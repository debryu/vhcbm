import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from matplotlib import pyplot as plt
from argus.utils.utils import set_seed
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from loguru import logger
from argus.models.base import BaseModel
import wandb
import numpy as np
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

def serialize_kernel_params(kernel):
    state = kernel.state_dict()
    return {k: v.detach().cpu().numpy().tolist() for k, v in state.items()}

class LatentJitterKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_latents, **kwargs):
        super().__init__(**kwargs)
        # Register a learnable noise parameter for each latent/class
        self.register_parameter(
            name="raw_noise", 
            parameter=torch.nn.Parameter(torch.zeros(num_latents, 1))
        )
        self.register_constraint("raw_noise", gpytorch.constraints.Positive())

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return self.noise.expand(x1.shape[:-1])
        if torch.equal(x1, x2):
            # Add noise to the diagonal of the covariance matrix
            return torch.diag_embed(self.noise.expand(x1.shape[:-1]))
        return torch.zeros(x1.shape[:-1] + (x2.shape[-2],), device=x1.device)
    
class CosineLinearKernelBias(Kernel):
    @property
    def is_stationary(self):
        return False

    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        
    def _augment(self, x):
        # Append a constant bias term = 1
        ones = torch.ones(
            x.shape[:-1] + (1,),
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, ones], dim=-1)

    def forward(self, x1, x2, diag=False, **params):
        # Augment inputs
        x1 = self._augment(x1)
        x2 = self._augment(x2)

        # Normalize to unit norm
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if diag:
        # x1_norm.shape[:-1] captures all batch dims and the 'n' dimension.
            return torch.ones(
                x1_norm.shape[:-1], 
                device=x1_norm.device, 
                dtype=x1_norm.dtype
            )

        # Cosine similarity in augmented space
        return x1_norm @ x2_norm.transpose(-2, -1)
    
class LinearWithBiasKernel(Kernel):
    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, has_lengthscale=False, **kwargs)
    
        # Weight variance σ_w^2
        self.register_parameter(
            name="raw_weight_variance",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_weight_variance", Positive())

        # Bias variance σ_b^2
        self.register_parameter(
            name="raw_bias_variance",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_bias_variance", Positive())

    @property
    def weight_variance(self):
        return self.raw_weight_variance_constraint.transform(self.raw_weight_variance)  # type:ignore

    @property
    def bias_variance(self):
        return self.raw_bias_variance_constraint.transform(self.raw_bias_variance)   # type:ignore

    def forward(self, x1, x2, diag=False, **params):
        # Normalize to unit norm
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if diag:
            # For diagonal covariance, each point with itself = 1
            return torch.ones(x1.size(0), device=x1.device, dtype=x1.dtype)
        
        # Dot product term
        dot_term = x1_norm @ x2_norm.transpose(-2, -1)

        # Apply weight variance
        K = self.weight_variance * dot_term + self.bias_variance

        # Still have to fix diag!!!
        if diag:
            return K.diag()
        return K

class CosineLinearKernel(Kernel):
    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
    
    @property    
    def is_stationary(self):
        # Cosine similarity is not stationary (depends on absolute orientation)
        return False

    def forward(self, x1, x2, diag=False, **params):
        # Normalize to unit norm
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if diag:
            # batch_shape = x1.shape[:-2],  num_points = x1.shape[-2]
            return torch.ones(
                *x1.shape[:-1],  # this gives [5, 84]
                device=x1.device,
                dtype=x1.dtype
            )

        
        # Cosine similarity = dot product of unit vectors
        return x1_norm @ x2_norm.transpose(-2, -1)
    
class RBFPlusCosineLinearKernel(gpytorch.kernels.Kernel):
    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.cosine = gpytorch.kernels.ScaleKernel(CosineLinearKernel())
        # Learnable weights
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(1)))
        
    def forward(self, x1, x2, diag=False, **params):
        alpha = torch.nn.functional.softplus(self.raw_alpha)  # positivity constraint    # type:ignore
        beta = torch.nn.functional.softplus(self.raw_beta)   # type:ignore
        return alpha * self.rbf(x1, x2, diag=diag, **params) \
             + beta * self.cosine(x1, x2, diag=diag, **params)



class MCSVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, kernel, learn_mean, num_classes, num_latents = 12):
            '''
            kernel=(RBF,lin,poly)
            '''
            inducing_points = inducing_points.unsqueeze(0).repeat(num_latents,1,1)
            #print(inducing_points.shape)
            
            self.n_inducing_points = inducing_points.size(1)
            # Variational distribution and strategy with inducing points
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=num_classes,
                num_latents=num_latents,
                latent_dim=-1
            )
            
            super().__init__(variational_strategy)

            # Define mean and kernel
            if learn_mean:
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            else:
                self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
            if kernel=='RBF':
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents]))
            elif kernel=='matern':
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents]))
            elif kernel == 'RBFn':
                # 1. The Signal (has its own scale)
                self.signal_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
                    batch_shape=torch.Size([num_latents])
                )
                
                # 2. The Jitter (independent of the signal scale)
                self.jitter_kernel = LatentJitterKernel(num_latents=num_latents)
                
                # 3. Sum them OUTSIDE the ScaleKernel
                self.covar_module = self.signal_kernel + self.jitter_kernel
            elif kernel == 'maternn':
                self.signal_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents]))
                
                # 2. The Jitter (independent of the signal scale)
                self.jitter_kernel = LatentJitterKernel(num_latents=num_latents)
                
                # 3. Sum them OUTSIDE the ScaleKernel
                self.covar_module = self.signal_kernel + self.jitter_kernel
            elif kernel == 'lin':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.LinearKernel(batch_shape=torch.Size([num_latents])),
                    batch_shape=torch.Size([num_latents])
                )
                # Inside MCSVGPModel.__init__
            elif kernel == 'poly':
                # Polynomial kernel: (x1'x2 + offset)^degree
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.PolynomialKernel(
                        power=2, # Start with degree 2
                        batch_shape=torch.Size([num_latents])
                    ),
                    batch_shape=torch.Size([num_latents])
                )
            elif kernel == 'cos':
                self.covar_module = gpytorch.kernels.ScaleKernel(CosineLinearKernel(batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents]))
            elif kernel == 'cosb':
                self.covar_module = gpytorch.kernels.ScaleKernel(CosineLinearKernelBias(batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents]))
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class MCSVGP(BaseModel):
    def __init__(self, args, concept_structure):
        super().__init__(self)
        self.balanced = args.balanced
        self.kernel = args.kernel
        self.masking = args.masking
        self.num_concepts = args.num_concepts
        self.num_one_hot_concepts = len(concept_structure)
        self.learn_mean = args.learn_mean
        self.lengthscale = args.lengthscale
        self.outputscale = args.outputscale
        self.gp_training_iter = args.gp_training_iter
        self.inputnoise = args.inputnoise
        self.seed = args.seed
        self.device = args.device
        self.lr = args.lr
        self.num_latents = args.num_latents
        self.num_likelihood_samples = args.num_likelihood_samples
        self.args = args
        self.learnnoise = args.learnnoise
        self.learnoutputscale = args.learnoutputscale
        self.learnlengthscale = args.learnlengthscale
        self.concept_structure = concept_structure
        
        '''
        concept_structure_example =    {'concept1':(0,9),
                                        'concept2':(10,19),
                                        'concept3':(20,29),
                                        'concept4':(30,37),
                                        'concept4':(38,41),
                                        }
        '''
        
        # Set random seed for reproducibility
        set_seed(self.seed)
        #gpytorch.settings.cholesky_jitter(float=1e-4, double=1e-5)
        
        # Number of inducing points is the 1 dimension of the tensor
        self.n_inducing_points = args.n_inducing_points

        logger.debug(f'Find inducing points of size {len(args.inducing_points)}')
        # Run K-means clustering to find inducing point centers
        #kmeans = KMeans(n_clusters=self.n_inducing_points).fit(args.inducing_points.cpu().numpy())
        kmeans = MiniBatchKMeans(n_clusters=self.n_inducing_points, batch_size=1024, n_init=1).fit(args.inducing_points.cpu().numpy())
        inducing_points_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(self.device)
        idx = self.training_pool
        inducing_points_init = self.args.inducing_points[idx].to(self.device)
        X = self.args.inducing_points
        N = X.shape[0]
        idx = torch.randperm(N)[:self.n_inducing_points]
        inducing_points_init = self.args.inducing_points[idx].to(self.device)
        #X = args.inducing_points
        #N = X.shape[0]
        #idx = torch.randperm(N)[:self.n_inducing_points]
        #inducing_points_init = args.inducing_points[idx].to(self.device)
        #print(inducing_points_init.shape)
    
        #inducing_points_init = torch.empty(self.n_inducing_points, 768).uniform_(-1, 1)    
        
        self.likelihoods = []
        self.models = []
        logger.debug('Creating model')
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            if num_classes == 1:
                num_classes = 2
            ind_point = inducing_points_init
            
           
            # Initialize model and independent Bernoulli likelihood
            #print(num_classes)
            model = MCSVGPModel(ind_point, self.kernel, self.learn_mean, num_classes=num_classes, num_latents=self.args.num_latents)
            model.to(self.device)
            if self.kernel == 'RBF':
                model.covar_module.base_kernel.initialize(lengthscale=self.lengthscale) # type:ignore
                model.covar_module.initialize(outputscale=self.outputscale)
                model.covar_module.base_kernel.raw_lengthscale.requires_grad = True  # type:ignore
                model.covar_module.raw_outputscale.requires_grad = True  # type:ignore
                
            # Set likelihood 
            #likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
            forced_y = torch.tensor(range(num_classes))
            likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=forced_y, 
                num_classes=num_classes,
                learn_additional_noise=True,
                batch_shape=torch.Size([num_classes])
            )
            likelihood = likelihood.to(self.device)
            #likelihood.noise = self.inputnoise
            
            if self.kernel == 'RBF':
                if not self.learnlengthscale:
                    try:
                        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False     # type:ignore
                    except:
                        logger.error("Failed to block lengthscale gradient updates")
                if not self.learnoutputscale:
                    try:
                        model.covar_module.raw_outputscale.requires_grad = False     # type:ignore
                    except:
                        logger.error("Failed to block outputscale gradient updates")
                if not self.learnnoise:
                    try:
                        likelihood.noise_covar.raw_noise.requires_grad = False   # type:ignore
                    except:
                        logger.error("Failed to block noise gradient updates. There is no noise modeling part in this version")
            
                
            self.likelihoods.append(likelihood)
            self.models.append(model)

    def get_probs(self,predictions :torch.Tensor) -> torch.Tensor:
        output = torch.zeros(predictions.shape)
        for concept_name, (start_idx, end_idx) in self.concept_structure.items():
            output[:,start_idx:end_idx+1] = torch.nn.functional.softmax(predictions[:,start_idx:end_idx+1], dim=1)
        return output

    def reset_models(self):
        pool_tensor = torch.as_tensor(self.training_pool, device=self.device)
        #kmeans = MiniBatchKMeans(n_clusters=self.n_inducing_points, batch_size=1024, n_init=1).fit(self.args.inducing_points.cpu().numpy())
        #inducing_points_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(self.device)
        #inducing_points_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(self.device)
        
        self.likelihoods = []
        self.models = []
        logger.debug('Creating model')
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            if self.masking:
                idx_to_keep = torch.where(self.mask[:,task_id] == 0)[0].to(self.device)
                mask_in_pool = torch.isin(pool_tensor, idx_to_keep)
                rows_to_grab = torch.where(mask_in_pool)[0]
                self.n_inducing_points = len(rows_to_grab)
                self.n_inducing_points = min(len(rows_to_grab), 500)
            else:
                self.n_inducing_points = len(self.training_pool)

            num_classes = len(range(idx_range[0],idx_range[1]+1))
            if num_classes == 1:
                num_classes = 2
            # Initialize model and independent Bernoulli likelihood
            logger.debug(f"Initialized with {self.n_inducing_points} inducing points")
            X = self.args.inducing_points
            N = X.shape[0]
            idx = torch.randperm(N)[:self.n_inducing_points]
            inducing_points_init = self.args.inducing_points[idx].to(self.device)
            model = MCSVGPModel(inducing_points_init, self.kernel, self.learn_mean, num_classes=num_classes, num_latents=self.args.num_latents)
            model.to(self.device)
            Z = model.variational_strategy.base_variational_strategy.inducing_points.detach().cpu()
            logger.warning(f"Inducing points shape: {Z.shape}")
            if self.kernel == 'RBF':
                model.covar_module.base_kernel.initialize(lengthscale=self.lengthscale) # type:ignore
                model.covar_module.initialize(outputscale=self.outputscale)
            # Set likelihood 
            likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
            likelihood = likelihood.to(self.device)
            likelihood.noise = self.inputnoise
            if not self.learnnoise:
                likelihood.noise_covar.raw_noise.requires_grad = False
            else:
                logger.warning(f"Learning noise")
            self.likelihoods.append(likelihood)
            self.models.append(model)
            
    def get_train_freqs(self, dataset_class) -> list[float]:
        """
        Returns a list of length n_concepts, each element is a float.
        """
        activations = []
        for sample_id in self.training_pool:
            for c_id in range(self.num_concepts):
                _, concepts,label = dataset_class[sample_id]
                activations.append(concepts)
        activations = torch.stack(activations, dim=0)
        n_samples = activations.shape[0]
        occ = torch.sum(activations, dim=0)
        return ((100*occ)/n_samples).tolist()
            

    def get_params(self):
        params = []
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            logger.warning(f"Task: {task_id+1}")
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            if num_classes == 1:
                num_classes = 2
            params_dict = {}
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            model.eval()
            likelihood.eval()

            # ---- 1. Kernel parameters ----
            if isinstance(model.covar_module, gpytorch.kernels.AdditiveKernel):
                for iii,k in enumerate(model.covar_module.kernels):  # List of sub-kernels
                    if hasattr(k, "base_kernel") and hasattr(k.base_kernel, "lengthscale"):
                        logger.warning(f"Lengthscale: {k.base_kernel.lengthscale}") # type:ignore
                        try:
                            params_dict.update({f"c{task_id}-lengthscale {iii}":k.base_kernel.lengthscale.detach().cpu().numpy()})  # type:ignore
                        except:
                            params_dict.update({f"c{task_id}-lengthscale {iii}":'err'})
            elif hasattr(model.covar_module.base_kernel, "lengthscale") and model.covar_module.base_kernel.lengthscale is not None:
                logger.warning(f"Lengthscale: {model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")
                params_dict.update({f"c{task_id}-lengthscale": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()})
            else:
                pass
            if hasattr(model.covar_module, "outputscale")and model.covar_module.outputscale is not None:
                logger.warning(f"Outputscale: {model.covar_module.outputscale.detach().cpu().numpy()}") # type:ignore
                params_dict.update({f"c{task_id}-outputscale":model.covar_module.outputscale.detach().cpu().numpy()})   # type:ignore

            # ---- 3. Inducing point locations ----
            Z = model.variational_strategy.base_variational_strategy.inducing_points.detach().cpu()
            logger.warning(f"Inducing points shape: {Z.shape}")

            # ---- 4. Likelihood noise ----
            #if hasattr(likelihood, "noise"):
            #    logger.warning(f"Noise: {likelihood.noise}")
               
            try:
                test = model.covar_module
                test = serialize_kernel_params(test)
            except:      
                pass      
            params.append(test)
                
        return params

    def train(self, dataset,labels, patience = np.inf, text_embeddings = None):
        #print(self.get_params())
        concept_used = 0
        pool_tensor = torch.as_tensor(self.training_pool, device=self.device)
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            if self.masking:
                rows_to_grab = torch.where(self.mask[:,task_id] == 0)[0].to(self.device)
                #random_indices = torch.randint(0, len(labels), (250,), device=self.device)
                ## 2. Concatenate with your existing active indices
                #combined = torch.cat([rows_to_grab, random_indices,torch.tensor(self.training_pool).long().to(self.device)])
                ## 3. Use unique to remove duplicates and sort the result
                ## Sorting is highly recommended for memory efficiency in GPyTorch
                #final_indices = torch.unique(combined)
                final_indices = torch.unique(rows_to_grab)
            else:
                rows_to_grab = range(len(dataset))
            #print(self.masking)
            #if self.masking:
            #    idx_to_keep = torch.where(self.mask[:,task_id] == 0)[0]
            #    allowed_set = set(idx_to_keep.tolist())
            #    rows_to_grab = [i for i, global_id in enumerate(self.training_pool) if global_id in allowed_set]
            #else:
            #    rows_to_grab = range(len(dataset))
            concept_used += len(rows_to_grab)
            _train_DS = dataset[final_indices]
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            logger.info(f"[{len(self.training_pool)} - {len(rows_to_grab)}] Training {task_id+1}/{len(self.concept_structure)}")
            model = self.models[task_id]
            model.to(self.device)
            
            # Handle binary (2 classes) concepts that represent the concept in a single dimension (0-1)
            if num_classes == 1:
                train_y = labels[final_indices, idx_range[0]]
                freq_y = labels[rows_to_grab, idx_range[0]]
                num_classes = 2
            else:
                train_y = torch.argmax(labels[final_indices, idx_range[0] : idx_range[1]+1], dim=1)
                freq_y = torch.argmax(labels[rows_to_grab, idx_range[0] : idx_range[1]+1], dim=1)
            
            
            # If a class is missing in the train set, the likelihood will have wrong shape
            # Since it uses the maximum value in the train classes to choose num of classes,
            # add a sample at the end with the correct num of classes
            fake_label = [num_classes-1]
            forced_y = torch.cat([train_y, torch.tensor(fake_label).long().to(self.device)])
            likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=forced_y, 
                num_classes=num_classes,
                learn_additional_noise=True,
                batch_shape=torch.Size([num_classes])
            )
            likelihood.to(self.device)
            # Remove the last sample added 
            likelihood.transformed_targets = likelihood.transformed_targets[:,0:-1]
            # Store the likelihood
            self.likelihoods[task_id] = likelihood

            #print(transformed_targets)
            #asd
            # The 'noise' parameter in Dirichlet is actually inside the noise_covar of each task
            likelihood.initialize(noise=torch.full((num_classes, 1), 0.01))            
            likelihood.train()
            model.train()
            
            #mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=dataset.shape[0])
            #mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()}
            ], lr=self.lr)
            scheduler = ReduceLROnPlateau(optimizer, patience=80, factor=0.8)
            # Patience was 18
            # Training loop
            
            best_loss = 10000000
            p = 0
            for i in range(self.gp_training_iter):
                optimizer.zero_grad()
                #print(dataset)
                output = model(_train_DS)
                mean = output.mean.transpose(-1, -2)
                variances = output.variance.transpose(-1, -2)
                #print(mean.shape)
                #print(variances.shape)
                batched_output = gpytorch.distributions.MultivariateNormal(
                    mean, 
                    torch.diag_embed(variances)
                )
                
                #print(f"GP Output Batch Shape: {output.batch_shape}") # Should be [10]
                #print(f"Likelihood Noise Shape: {likelihood.noise.shape}") # Should be [10, 1]
                
                #preds = likelihood(output)  
                
            

                # Calculate frequencies from the original train_y (before padding)
                class_counts = torch.bincount(freq_y, minlength=num_classes).float()
                safe_counts = torch.clamp(class_counts, min=1.0)
                class_weights = freq_y.size(0) / (num_classes * safe_counts)
                per_sample_log_prob = likelihood.expected_log_prob(likelihood.transformed_targets, batched_output)
                
                if self.balanced:
                    per_sample_log_prob = (per_sample_log_prob.T * class_weights).sum(1)
                else:
                    per_sample_log_prob = per_sample_log_prob.sum(0)
                loss_mask = torch.isin(final_indices, rows_to_grab).float().to(self.device)

                masked_log_likelihood = (per_sample_log_prob * loss_mask)
                #loss = -mll(batched_output, likelihood.transformed_targets)  # type:ignore
                loss = masked_log_likelihood / (loss_mask.sum() + 1e-08)
                
                kl_term = model.variational_strategy.kl_divergence()
                
                # subtracting the num of classes since the min loss of constraint is num_classes
                loss = loss.mean() - kl_term*(i*1e-04)*1e-05
                
                loss = -loss
                #print(f"Orth loss {torch.sum(constraint)}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=1.0)
                loss.backward()
                current_lr = optimizer.param_groups[0]['lr']
                if (i + 1) % 500 == 0:
                    
                    logger.debug(f'Iter {i+1}/{self.gp_training_iter} - Loss: {loss.item():.3f} - LR: {current_lr} - C:{loss_mask.sum()} |{kl_term}')
                    #latent_dist = model(_train_DS)
                    #noise = likelihood.noise_covar.noise.detach().cpu()
                    
                    
                optimizer.step()
                scheduler.step(loss.detach().cpu())
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    p=0
                else:
                    p+=1
                
                if current_lr < 1e-07:
                    logger.warning(f"Stopped early due to lr approaching 0")
                    break
                if p > patience:
                    logger.warning(f"Stopped early due to no improvement after {patience} iters.")
                    break
            model.eval()
            self.concept_used = concept_used
    
    def eval(self, dataset):
        pred_probs = []
        pred_vars = []
        soft_pred_probs = []
        soft_pred_vars = []
        latent_uncs = []
        latent_concs = []
        predictions = []
        mc_predictions = []
        
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            if num_classes == 1:
                softmax_likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=2, mixing_weights=False)
            else:
                softmax_likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
            likelihood.eval()
            model.eval()
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
            batch_pred_probs = []
            batch_pred_vars = []
            batch_soft_pred_probs = []
            batch_soft_pred_vars = []
            batch_predictions = []
            batch_probabilities = []
            batch_uncs = []
            batch_latent_conc = []
            evidence = []
            for batch in loader:
                '''
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    latent_dist = model(batch)
                    mean = latent_dist.mean.transpose(-1, -2)
                    covar = torch.diag_embed(latent_dist.variance.transpose(-1, -2))
                    batched_dist = gpytorch.distributions.MultivariateNormal(mean, covar) 
                    dist = likelihood(batched_dist)    
                    probs = dist.mean
                    #print(probs[:,0])
                    uncertainty = dist.variance
                    
                    batch_uncs.append(uncertainty)
                    batch_probabilities.append(probs)
                   
                    #print(probs.shape)
                    #print(uncertainty.shape)
                '''
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                    latent_dist = model(batch)
                    softmax_preds = softmax_likelihood(latent_dist)
                    mean = latent_dist.mean.transpose(-1, -2)
                    variances = latent_dist.variance.transpose(-1, -2)
                    #print(mean.shape)
                    #print(variances.shape)
                    batched_output = gpytorch.distributions.MultivariateNormal(
                        mean, 
                        torch.diag_embed(variances)
                    )
                    preds = likelihood(batched_output) 
                    #print(preds.probs.mean(dim=0).detach().cpu()[0])
                    if num_classes == 1:
                        #print(preds.probs.mean(dim=0))
                        batch_pred_probs.append(preds.mean.detach().cpu().permute(1,0)[:,1].unsqueeze(-1))  
                        batch_pred_vars.append(preds.stddev.detach().cpu().permute(1,0)[:,1].unsqueeze(-1)) 
                        batch_soft_pred_probs.append(softmax_preds.probs.mean(dim=0).detach().cpu()[:,1].unsqueeze(-1))  
                        batch_soft_pred_vars.append(softmax_preds.probs.std(dim=0).detach().cpu()[:,1].unsqueeze(-1))   
                    else:
                        batch_pred_probs.append(preds.mean.detach().cpu().permute(1,0))   # Average on mc samples
                        batch_pred_vars.append(preds.stddev.detach().cpu().permute(1,0))  
                        batch_soft_pred_probs.append(softmax_preds.probs.mean(dim=0).detach().cpu())   # Average on mc samples
                        batch_soft_pred_vars.append(softmax_preds.probs.std(dim=0).detach().cpu())
                    
                    latent_means = latent_dist.mean.transpose(-1, -2)
                    # Get Concentration [classes, batch]
                    concentration = torch.exp(latent_means)
                    batch_latent_conc.append(concentration)
                    # Total evidence (precision) for each sample
                    total_evidence = concentration.sum(dim=-1)

                    #print(task_name)
                    #print(self.get_latent_stats(model,batch))
            soft_pred_probs.append(torch.cat(batch_soft_pred_probs, dim=0))
            soft_pred_vars.append(torch.cat(batch_soft_pred_vars, dim=0))
            pred_probs.append(torch.cat(batch_pred_probs, dim=0))
            pred_vars.append(torch.cat(batch_pred_vars, dim=0))
            # Get predictions [N]
            if num_classes == 1:
                prediction = (torch.cat(batch_pred_probs, dim=0) > 0.5).long()
                mc_predictions.append(prediction)
            else:
                prediction = torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1)
                mc_predictions.append(torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1))
                prediction = torch.nn.functional.one_hot(prediction, num_classes=num_classes)
            
            batch_predictions.append(prediction)
            # shape [n_classes, n_samples]
            #print(torch.cat(batch_uncs, dim=1).shape)
            #latent_uncs.append(torch.cat(batch_uncs, dim=1))
            latent_concs.append(torch.cat(batch_latent_conc, dim=1))
            predictions.append(torch.cat(batch_predictions, dim=0))
        
       
        soft_pred_probs = torch.cat(soft_pred_probs,dim=1)
        soft_pred_vars = torch.cat(soft_pred_vars,dim=1)
        pred_probs = torch.cat(pred_probs,dim=1)
        #print(pred_probs.shape)
        #print(predictions[0].shape)
        pred_vars = torch.cat(pred_vars,dim=1)
        predictions = torch.cat(predictions, dim=1)
        #print(mc_predictions)
        mc_predictions = torch.cat(mc_predictions, dim=0)
        #print(mc_predictions.shape)
        latent_concs = torch.cat(latent_concs, dim=0).transpose(1,0)
       
        #print("preds", predictions.shape)
        #print("conc", latent_concs.shape)
        #print('probs', pred_probs.shape)
        #print(pred_probs[0])
        #print(pred_vars[0])
        return {'means':pred_probs, 'std':pred_vars, 'preds':predictions, 'concentration':latent_concs, 'predsMC':mc_predictions, 'probs':soft_pred_probs, 'soft_std':soft_pred_vars}
    
    def evalmc(self, dataset):
        pred_probs = []
        pred_vars = []
        soft_pred_probs = []
        soft_pred_vars = []
        latent_uncs = []
        latent_concs = []
        predictions = []
        mc_predictions = []
        
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            if num_classes == 1:
                softmax_likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=2, mixing_weights=False)
            else:
                softmax_likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
            likelihood.eval()
            model.eval()
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
            batch_pred_probs = []
            batch_pred_vars = []
            batch_soft_pred_probs = []
            batch_soft_pred_vars = []
            batch_predictions = []
            batch_probabilities = []
            batch_uncs = []
            batch_latent_conc = []
            evidence = []
            for batch in loader:
                '''
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    latent_dist = model(batch)
                    mean = latent_dist.mean.transpose(-1, -2)
                    covar = torch.diag_embed(latent_dist.variance.transpose(-1, -2))
                    batched_dist = gpytorch.distributions.MultivariateNormal(mean, covar) 
                    dist = likelihood(batched_dist)    
                    probs = dist.mean
                    #print(probs[:,0])
                    uncertainty = dist.variance
                    
                    batch_uncs.append(uncertainty)
                    batch_probabilities.append(probs)
                   
                    #print(probs.shape)
                    #print(uncertainty.shape)
                '''
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                    latent_dist = model(batch)
                    softmax_preds = softmax_likelihood(latent_dist)
                    mean = latent_dist.mean.transpose(-1, -2)
                    variances = latent_dist.variance.transpose(-1, -2)
                    #print(mean.shape)
                    #print(variances.shape)
                    batched_output = gpytorch.distributions.MultivariateNormal(
                        mean, 
                        torch.diag_embed(variances)
                    )
                    preds = likelihood(batched_output) 
                    #print(preds.probs.mean(dim=0).detach().cpu()[0])
                    if num_classes == 1:
                        #print(preds.probs.mean(dim=0))
                        batch_pred_probs.append(preds.mean.detach().cpu().permute(1,0))  
                        batch_pred_vars.append(preds.stddev.detach().cpu().permute(1,0)) 
                        batch_soft_pred_probs.append(softmax_preds.probs.mean(dim=0).detach().cpu())  
                        batch_soft_pred_vars.append(softmax_preds.probs.std(dim=0).detach().cpu())   
                    else:
                        batch_pred_probs.append(preds.mean.detach().cpu().permute(1,0))   # Average on mc samples
                        batch_pred_vars.append(preds.stddev.detach().cpu().permute(1,0))  
                        batch_soft_pred_probs.append(softmax_preds.probs.mean(dim=0).detach().cpu())   # Average on mc samples
                        batch_soft_pred_vars.append(softmax_preds.probs.std(dim=0).detach().cpu())
                    
                    latent_means = latent_dist.mean.transpose(-1, -2)
                    # Get Concentration [classes, batch]
                    concentration = torch.exp(latent_means)
                    batch_latent_conc.append(concentration)
                    # Total evidence (precision) for each sample
                    total_evidence = concentration.sum(dim=-1)

                    #print(task_name)
                    #print(self.get_latent_stats(model,batch))
            soft_pred_probs.append(torch.cat(batch_soft_pred_probs, dim=0))
            #print(torch.argmax(soft_pred_probs[-1], dim=1)[0:10])
            #akak = soft_pred_probs[-1]
            #akakk = torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1)
            #print(akakk[0:10])
            soft_pred_vars.append(torch.cat(batch_soft_pred_vars, dim=0))
            pred_probs.append(torch.cat(batch_pred_probs, dim=0))
            pred_vars.append(torch.cat(batch_pred_vars, dim=0))
            
            # Get predictions [N]
            if num_classes == 1:
                mc_predictions.append(torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1))
                prediction = torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1).unsqueeze(-1)
            else:
                #prediction = (torch.cat(batch_pred_probs, dim=0) > 0.5).long()
                mc_predictions.append(torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1))
                #prediction = torch.nn.functional.one_hot(prediction, num_classes=num_classes)
                prediction = torch.nn.functional.one_hot(torch.argmax(torch.cat(batch_pred_probs, dim=0),dim=1), num_classes=num_classes)
            
            
            #batch_predictions.append(prediction)
            # shape [n_classes, n_samples]
            #print(torch.cat(batch_uncs, dim=1).shape)
            #latent_uncs.append(torch.cat(batch_uncs, dim=1))
            latent_concs.append(torch.cat(batch_latent_conc, dim=1))
            predictions.append(prediction)
            #print(torch.cat(batch_predictions, dim=0).shape)
            
        
        
        soft_pred_probs = torch.cat(soft_pred_probs,dim=1)
        soft_pred_vars = torch.cat(soft_pred_vars,dim=1)
        pred_probs = torch.cat(pred_probs,dim=1)
        #print(pred_probs.shape)
        #print(predictions[0].shape)
        pred_vars = torch.cat(pred_vars,dim=1)
        
        predictions = torch.cat(predictions, dim=1)
        #print(mc_predictions)
        mc_predictions = torch.cat(mc_predictions, dim=0)
        #print(mc_predictions.shape)
        latent_concs = torch.cat(latent_concs, dim=0).transpose(1,0)
        
        #print("preds", predictions.shape)
        #print("conc", latent_concs.shape)
        #print('probs', pred_probs.shape)
        #print(pred_probs[0])
        #print(pred_vars[0])
        return {'means':pred_probs, 'std':pred_vars, 'preds':predictions, 'concentration':latent_concs, 'predsMC':mc_predictions, 'probs':soft_pred_probs, 'soft_std':soft_pred_vars}
    
    def unc_score(self, dataset) -> torch.Tensor:
        '''
        Returns mean:torch.Tensor
        '''
        scores = []
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            if num_classes < 2:
                num_classes = 2
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            likelihood.eval()
            model.eval()
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
            batch_pred_probs = []
            batch_pred_vars = []
            for batch in loader:
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                    preds = likelihood(model(batch))              
                    batch_pred_probs.append(preds.probs.mean(dim=0).detach().cpu())   # Average on mc samples
                    batch_pred_vars.append(preds.probs.std(dim=0).detach().cpu())
                    
            batch_pred_probs = torch.cat(batch_pred_probs, dim=0)
            
            scores.append(torch.prod(batch_pred_probs, dim=1)*(num_classes**num_classes))
            #entr = self.shannon_entropy(torch.cat(batch_pred_probs, dim=0))
            
        return torch.stack(scores, dim=1)
    
    
    def mean(self, dataset) -> torch.Tensor:
        '''
        Returns mean:torch.Tensor
        '''
        pred_probs = []
        pred_vars = []
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            likelihood.eval()
            model.eval()
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
            batch_pred_probs = []
            batch_pred_vars = []
            for batch in loader:
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                    preds = likelihood(model(batch))              
                    batch_pred_probs.append(preds.probs.mean(dim=0).detach().cpu())   # Average on mc samples
                    batch_pred_vars.append(preds.probs.std(dim=0).detach().cpu())
            pred_probs.append(torch.cat(batch_pred_probs, dim=0))
            pred_vars.append(torch.cat(batch_pred_vars, dim=0))
        pred_probs = torch.cat(pred_probs,dim=1)
        return pred_probs
    
    
    def probs(self, dataset) -> torch.Tensor:
        '''
        Returns mean:torch.Tensor
        '''
        pred_probs = []
        for task_id in range(self.num_concepts):
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            likelihood.eval()
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(32):
                preds = likelihood(model(dataset).rsample(torch.Size([50])))  
                pred_probs.append(preds.probs)   # Average on mc samples
        pred_probs = torch.stack(pred_probs,dim=1)
        return pred_probs
    
    def shannon_entropy(self, probs):
        unc = []
        for i in range(probs.shape[1]):
            unc.append(probs[:,i]*(1-probs[:,i]))
        unc = torch.stack(unc, dim=1)
        return unc
    
    def get_mean_per_sample_per_class(self, model, latent_means):
        # 1. Get W: [5, 10] (latents x classes)
        W = model.variational_strategy.lmc_coefficients 
        
        mean_matrix = torch.matmul(W.t().cpu(), latent_means.cpu())
        
        return mean_matrix
    
    import torch


    
    def entropy(self, dataset) -> torch.Tensor:
        '''
        Returns entropy:torch.Tensor
        '''
        pred_probs = self.mean(dataset)
        entropy = - (pred_probs * pred_probs.log() + (1 - pred_probs) * (1 - pred_probs).log())
        entropy = entropy.nan_to_num()  # Clean up any NaNs from log(0)
        return entropy
    
    def latent_variance(self, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns tuple(logits, vars) 
        '''
        pred_probs = []
        pred_vars = []
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            likelihood.eval()
            model.eval()
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
            batch_pred_probs = []
            batch_pred_vars = []
            for batch in loader:
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                    preds = likelihood(model(batch))              
                    batch_pred_probs.append(preds.probs.mean(dim=0).detach().cpu())   # Average on mc samples
                    batch_pred_vars.append(preds.probs.std(dim=0).detach().cpu())
            pred_probs.append(torch.cat(batch_pred_probs, dim=0))
            pred_vars.append(torch.cat(batch_pred_vars, dim=0))
            #predictions = pred_probs[0].tolist()
            #label = labels[:,task_id]
            #for a,b in zip(predictions,label):
            #    print(a,b)
        
        pred_probs = torch.cat(pred_probs,dim=1)
        pred_vars = torch.cat(pred_vars,dim=1)
        #print('std',pred_vars)
        return pred_probs, pred_vars
    