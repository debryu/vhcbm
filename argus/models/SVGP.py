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
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

def serialize_kernel_params(kernel):
    state = kernel.state_dict()
    return {k: v.detach().cpu().numpy().tolist() for k, v in state.items()}


class LabelKernel(gpytorch.kernels.Kernel):
    """
    A categorical kernel that only correlates points with the same label.
    K_ij = label_variance if y_i == y_j else 0
    """
    @property
    def is_stationary(self):
        return False

    def __init__(self):
        super().__init__()
        self.register_parameter(
            name="raw_label_variance",
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )
        self.register_constraint("raw_label_variance", gpytorch.constraints.Positive())

    @property
    def label_variance(self):
        return self.raw_label_variance_constraint.transform(self.raw_label_variance)

    def forward(self, x1, x2, diag=False, **params):
        """
        Assumes x1 and x2 are integer labels with shape [N, 1].
        """
        if diag:
            return torch.full(
                (x1.size(0),),
                self.label_variance,
                dtype=x1.dtype,
                device=x1.device
            )

        # Broadcast comparison: returns [N1, N2] boolean mask
        same_label = (x1 == x2.transpose(-2, -1)).float()
        return self.label_variance * same_label

class CosinePlusLabelKernel(gpytorch.kernels.Kernel):
    def __init__(self, alpha = -1, beta = -1):
        super().__init__()

        self.cosine = gpytorch.kernels.ScaleKernel(CosineLinearKernel())
        self.label = gpytorch.kernels.ScaleKernel(LabelKernel())
        if alpha == -1:
            # Mixing weights
            self.register_parameter(
                name="raw_alpha",
                parameter=torch.nn.Parameter(torch.zeros(1))
            )
            self.register_constraint("raw_alpha", gpytorch.constraints.Positive())
            self.alpha_constraint = None
        else:
            self.alpha_constraint = abs(alpha)
        if beta == -1:
            self.register_parameter(
                name="raw_beta",
                parameter=torch.nn.Parameter(torch.zeros(1))
            )
            self.register_constraint("raw_beta", gpytorch.constraints.Positive())
            self.beta_constraint = None
        else:
            self.beta_constraint = abs(alpha)
            
    @property
    def alpha(self):
        if self.alpha_constraint is not None:
            return self.alpha_constraint
        else:
            return self.raw_alpha_constraint.transform(self.raw_alpha)

    @property
    def beta(self):
        if self.beta_constraint is not None:
            return self.beta_constraint
        else:
            return self.raw_beta_constraint.transform(self.raw_beta)

    def forward(self, x1, x2, diag=False, **params):
        """
        x1, x2 must be tuples of (features, labels).
        """
        feats1, labels1 = x1
        feats2, labels2 = x2

        K_cos = self.cosine(feats1, feats2, diag=diag)
        K_lbl = self.label(labels1, labels2, diag=diag)

        return self.alpha * K_cos + self.beta * K_lbl


class LinearWithBiasKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)

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
        return self.raw_weight_variance_constraint.transform(self.raw_weight_variance)

    @property
    def bias_variance(self):
        return self.raw_bias_variance_constraint.transform(self.raw_bias_variance)

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

        if diag:
            return K.diag()
        return K

class CosineLinearKernelBias(Kernel):
    @property
    def is_stationary(self):
        return False

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
            return torch.ones(
                x1_norm.size(0),
                device=x1_norm.device,
                dtype=x1_norm.dtype,
            )

        # Cosine similarity in augmented space
        return x1_norm @ x2_norm.transpose(-2, -1)
    
class CosineLinearKernel(Kernel):
    @property
    def is_stationary(self):
        # Cosine similarity is not stationary (depends on absolute orientation)
        return False

    def forward(self, x1, x2, diag=False, **params):
        # Normalize to unit norm
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if diag:
            # For diagonal covariance, each point with itself = 1
            return torch.ones(x1.size(0), device=x1.device, dtype=x1.dtype)
        
        # Cosine similarity = dot product of unit vectors
        return x1_norm @ x2_norm.transpose(-2, -1)
    
class RBFNormKernel(gpytorch.kernels.Kernel):
    """
    RBF kernel applied to the L2 norm of the inputs.
    k(x, x') = exp(- (||x|| - ||x'||)^2 / (2 * l^2))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = gpytorch.kernels.RBFKernel(**kwargs)

    @property
    def is_stationary(self):
        return True

    def forward(self, x1, x2, diag=False, **params):
        # Compute norms
        x1_norm = x1.norm(dim=-1, keepdim=True)
        x2_norm = x2.norm(dim=-1, keepdim=True)

        return self.base_kernel(x1_norm, x2_norm, diag=diag, **params)
    
class RBFPlusCosineLinearKernel(gpytorch.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.cosine = gpytorch.kernels.ScaleKernel(CosineLinearKernel())
        # Learnable weights
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(1)))
        
    def forward(self, x1, x2, diag=False, **params):
        alpha = torch.nn.functional.softplus(self.raw_alpha)  # positivity constraint
        beta = torch.nn.functional.softplus(self.raw_beta)
        return alpha * self.rbf(x1, x2, diag=diag, **params) \
             + beta * self.cosine(x1, x2, diag=diag, **params)



class SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, kernel, learn_mean):
            '''
            kernel=(RBF,lin,poly)
            '''
            
            # Variational distribution and strategy with inducing points
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)

            # Define mean and kernel
            if learn_mean:
                self.mean_module = gpytorch.means.ConstantMean()
            else:
                self.mean_module = gpytorch.means.ZeroMean()
            if kernel=='RBF':
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            elif kernel == 'cos_y':
                self.covar_module1 = gpytorch.kernels.ScaleKernel(CosineLinearKernelBias())
                self.covar_module2 = LabelKernel()
            elif kernel == 'lin':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.LinearKernel()
                )
            elif kernel == 'poly':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.PolynomialKernel(power=3) 
                )
            elif kernel == 'last':
                self.covar_module = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.ProductKernel(
                                            CosineLinearKernelBias(),
                                            RBFNormKernel()
                                        )
                                    )
            elif kernel == 'sum':
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(CosineLinearKernelBias())
            elif kernel == 'cos':
                self.covar_module = gpytorch.kernels.ScaleKernel(CosineLinearKernel())
            elif kernel == 'cosb':
                self.covar_module = gpytorch.kernels.ScaleKernel(CosineLinearKernelBias())
            elif kernel == 'learnable':
                self.covar_module = gpytorch.kernels.ScaleKernel(RBFPlusCosineLinearKernel())
            elif kernel == 'nn':
                self.covar_module = gpytorch.kernels.ScaleKernel(LinearWithBiasKernel())
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
        def forward(self, x, labels = None):
            if labels is None:
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            else:
                raise NotImplementedError()
        
class SVGP(BaseModel):
    def __init__(self, args):
        super().__init__(self)
        self.kernel = args.kernel
        self.num_concepts = args.num_concepts
        self.learn_mean = args.learn_mean
        self.lengthscale = args.lengthscale
        self.outputscale = args.outputscale
        self.gp_training_iter = args.gp_training_iter
        self.inputnoise = args.inputnoise
        self.seed = args.seed
        self.device = args.device
        self.lr = args.lr
        self.num_likelihood_samples = args.num_likelihood_samples
        self.args = args
        self.learnnoise = args.learnnoise
        self.learnoutputscale = args.learnoutputscale
        self.learnlengthscale = args.learnlengthscale
        self.loss_history = []
        if hasattr(args,'masking'):
            self.masking = args.masking
            
        # Set random seed for reproducibility
        set_seed(self.seed)

        # Number of inducing points is the 1 dimension of the tensor
        self.n_inducing_points = args.n_inducing_points

        logger.debug(f'Find inducing points of size {len(args.inducing_points)}')
        # Run K-means clustering to find inducing point centers
        #kmeans = KMeans(n_clusters=self.n_inducing_points).fit(args.inducing_points.cpu().numpy())
        kmeans = MiniBatchKMeans(n_clusters=self.n_inducing_points, batch_size=1024, n_init=1).fit(args.inducing_points.cpu().numpy())
        inducing_points_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(self.device)
        #inducing_points_init = torch.empty(self.n_inducing_points, 768).uniform_(-1, 1)   
        # Random initialization
        # Experimental 
        X = args.inducing_points
        N = X.shape[0]
        idx = torch.randperm(N)[:self.n_inducing_points]
        inducing_points_init = args.inducing_points[idx].to(self.device)
        # end
        
        self.likelihoods = []
        self.models = []
        logger.debug('Creating model')
        for task_id in range(self.num_concepts):
            # Initialize model and independent Bernoulli likelihood
            model = SVGPModel(inducing_points_init, self.kernel, self.learn_mean)
            model.to(self.device)
            if self.kernel == 'RBF':
                model.covar_module.base_kernel.initialize(lengthscale=self.lengthscale) # type:ignore
                model.covar_module.initialize(outputscale=self.outputscale)
                model.covar_module.base_kernel.raw_lengthscale.requires_grad = True
                model.covar_module.raw_outputscale.requires_grad = True
                
            # Set likelihood 
            likelihood = gpytorch.likelihoods.BernoulliLikelihood()
            likelihood = likelihood.to(self.device)
            #likelihood.noise = self.inputnoise
            
            if self.kernel == 'RBF':
                if not self.learnlengthscale:
                    try:
                        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
                    except:
                        logger.error("Failed to block lengthscale gradient updates")
                if not self.learnoutputscale:
                    try:
                        model.covar_module.raw_outputscale.requires_grad = False
                    except:
                        logger.error("Failed to block outputscale gradient updates")
                if not self.learnnoise:
                    try:
                        likelihood.noise_covar.raw_noise.requires_grad = False
                    except:
                        logger.error("Failed to block noise gradient updates. There is no noise modeling part in this version")
                
            self.likelihoods.append(likelihood)
            self.models.append(model)

    def reset_models(self):
        kmeans = MiniBatchKMeans(n_clusters=self.n_inducing_points, batch_size=1024, n_init=1).fit(self.args.inducing_points.cpu().numpy())
        inducing_points_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(self.device)
        self.likelihoods = []
        self.models = []
        logger.debug('Creating model')
        for task_id in range(self.num_concepts):
            # Initialize model and independent Bernoulli likelihood
            model = SVGPModel(inducing_points_init, self.kernel, self.learn_mean)
            model.to(self.device)
            if self.kernel == 'RBF':
                model.covar_module.base_kernel.initialize(lengthscale=self.lengthscale) # type:ignore
                model.covar_module.initialize(outputscale=self.outputscale)
            # Set likelihood 
            likelihood = gpytorch.likelihoods.BernoulliLikelihood()
            likelihood = likelihood.to(self.device)
            likelihood.noise = self.inputnoise
            if not self.learnnoise:
                likelihood.noise_covar.raw_noise.requires_grad = False

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
        for task_id in range(self.num_concepts):
            params_dict = {}
            model = self.models[task_id]
            likelihood = self.likelihoods[task_id]
            model.eval()
            likelihood.eval()

            # ---- 1. Kernel parameters ----
            if isinstance(model.covar_module, gpytorch.kernels.AdditiveKernel):
                for iii,k in enumerate(model.covar_module.kernels):  # List of sub-kernels
                    if hasattr(k, "base_kernel") and hasattr(k.base_kernel, "lengthscale"):
                        logger.warning(f"Lengthscale: {k.base_kernel.lengthscale}")
                        try:
                            params_dict.update({f"c{task_id}-lengthscale {iii}":k.base_kernel.lengthscale.detach().cpu().numpy()})
                        except:
                            params_dict.update({f"c{task_id}-lengthscale {iii}":'err'})
            elif hasattr(model.covar_module.base_kernel, "lengthscale") and model.covar_module.base_kernel.lengthscale is not None:
                logger.warning(f"Lengthscale: {model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")
                params_dict.update({f"c{task_id}-lengthscale": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()})
            else:
                pass
            if hasattr(model.covar_module, "outputscale")and model.covar_module.outputscale is not None:
                logger.warning(f"Outputscale: {model.covar_module.outputscale.detach().cpu().numpy()}")
                params_dict.update({f"c{task_id}-outputscale":model.covar_module.outputscale.detach().cpu().numpy()})

            # ---- 3. Inducing point locations ----
            Z = model.variational_strategy.inducing_points.detach().cpu()
            logger.warning(f"Inducing points shape: {Z.shape}")

            # ---- 4. Likelihood noise ----
            if hasattr(likelihood, "noise"):
                logger.warning(f"Noise: {likelihood.noise}")
               
            try:
                test = model.covar_module
                test = serialize_kernel_params(test)
            except:      
                pass      
            params.append(test)
                
        return params

    def train(self, dataset,concept_labels, patience = np.inf, labels = None):
        self.loss_history = []
        concept_used = 0
        print(dataset.shape)
        print(concept_labels.shape)
        for task_id in range(self.num_concepts):
            # Mask the data
            label = concept_labels[:,task_id]
            idx_to_keep = torch.where(label != -1)
            label = label[idx_to_keep]
            _train_DS = dataset[idx_to_keep]
            concept_used += len(idx_to_keep[0])
            #
            
            logger.info(f"[{len(self.training_pool)}] Training {task_id+1}/{self.num_concepts}")
            model = self.models[task_id]
            model.to(self.device)
            likelihood = self.likelihoods[task_id]
            likelihood.to(self.device)
            likelihood.train()
            model.train()
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=_train_DS.shape[0])
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()}
            ], lr=self.lr)
            # Training loop
            
            best_loss = 10000000
            p = 0
            losses = []
            for i in range(self.gp_training_iter):
                optimizer.zero_grad()
                if self.kernel in ['cos_y']:
                    print("ds",_train_DS.shape)
                    print("labels", labels.shape)
                    output = model(inputs=_train_DS, labels=labels)
                else:
                    output = model(_train_DS)
                preds = likelihood(output)  
                #print(preds.mean)   # Average on mc samples
                #print(train_y[:,task_id].unsqueeze(-1).shape)
                
                loss = -mll(output, label.to(self.device))  # type:ignore
                loss = loss.mean()
                losses.append(loss.item())
                loss.backward()
                if (i + 1) % 500 == 0:
                    logger.debug(f'Iter {i+1}/{self.gp_training_iter} - Loss: {loss.item():.3f}')
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    p=0
                else:
                    p+=1
                if p > patience:
                    logger.warning(f"Stopped early due to no improvement after {patience} iters.")
                    break
            self.loss_history.append(losses)
            logger.warning(f"Concept used during training: {concept_used}")
            model.eval()
        logger.warning(f"Concept used during training: {concept_used}")
        self.concept_used = concept_used
        
    def eval(self, dataset, labels =None):
        pred_probs = []
        pred_vars = []
        latent_mean = []
        latent_var = []
        for task_id in range(self.num_concepts):
            model = self.models[task_id]
            with torch.no_grad():
                if self.kernel in ['cos_y']:
                    output = model(dataset, labels)
                else:
                    output = model(dataset)
                latent_mean.append(output.mean)
                latent_var.append(output.variance)
            likelihood = self.likelihoods[task_id]
            likelihood.eval()
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                preds = likelihood(model(dataset))  
                pred_probs.append(preds.mean)   # Average on mc samples
                pred_vars.append(preds.stddev)
            #predictions = pred_probs[0].tolist()
            #label = labels[:,task_id]
            #for a,b in zip(predictions,label):
            #    print(a,b)
        pred_probs = torch.stack(pred_probs,dim=1)
        pred_vars = torch.stack(pred_vars,dim=1)
        latent_mean = torch.stack(latent_mean,dim=1)
        latent_var = torch.stack(latent_var,dim=1)
        #print('std',pred_vars)
        return {'probs':pred_probs, 'std':pred_vars, 'latent_means':latent_mean, 'latent_vars':latent_var}
    
    def mean(self, dataset) -> torch.Tensor:
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
                preds = likelihood(model(dataset))  
                pred_probs.append(preds.mean)   # Average on mc samples
        pred_probs = torch.stack(pred_probs,dim=1)
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
        logits = []
        vars = []
        for task_id in range(self.num_concepts):
            model = self.models[task_id]
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = model(dataset) 
                logits.append(output.mean)
                vars.append(output.variance)
        logits = torch.stack(logits,dim=1)
        vars = torch.stack(vars,dim=1)
        return logits, vars
    
    def plot(self,dataset,labels):
        pred_probs = self.eval(dataset,labels)['probs']
        # Plot results for a few tasks
        plt.figure(figsize=(12, 6))
        for t in range(min(4, self.num_concepts)):
            plt.subplot(2, 2, t+1)
            plt.title(f"Task {t}")
            plt.plot(dataset.squeeze(), pred_probs[:, t].numpy(), label='Predictive Prob')
            plt.scatter(dataset.squeeze(), labels[:, t], c='k', s=10, label='Train Labels')
            plt.ylim([-0.1, 1.1])
            plt.legend()
        plt.tight_layout()
        plt.savefig("eval.png")
        
    def visualize2D(self, dataset, n=2):
        # Dimensionality Reduction on the embeddings
        from sklearn.decomposition import PCA
        X = PCA(n_components=n).fit_transform(dataset.cpu().numpy())
        
        
        # Plot predictions and uncertainty
        out_dict = self.eval(dataset)
        for ii in range(100):
            print(out_dict['probs'][ii], out_dict['std'][ii])
            asd
        
        res_dict = {}
        for i in range(self.num_concepts):
            # Create a side-by-side figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
            
            sc1 = axes[0].scatter(X[:, 0], X[:, 1], c=out_dict['probs'][:,i].cpu().numpy(), cmap='coolwarm', s=40)
            axes[0].set_title("GP Predictive Mean")
            axes[0].set_xlabel("Dim 1")
            axes[0].set_ylabel("Dim 2")
            fig.colorbar(sc1, ax=axes[0], label="Mean")

            sc2 = axes[1].scatter(X[:, 0], X[:, 1], c=out_dict['std'][:,i].cpu().numpy(), cmap='viridis', s=40)
            axes[1].set_title("GP Predictive Uncertainty (STD)")
            axes[1].set_xlabel("Dim 1")
            axes[1].set_ylabel("Dim 2")
            fig.colorbar(sc2, ax=axes[1], label="Variance")

            # Adjust layout and save
            plt.tight_layout()
            #plt.savefig(f"Concept_{i}.png", dpi=300, bbox_inches='tight')
            res_dict[f"Concept_{i}.png"] = wandb.Image(fig)
            plt.close()
        wandb.log(res_dict)
