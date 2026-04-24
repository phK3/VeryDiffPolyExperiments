import torch
import torch.nn.functional as F

class Adversary:

    def __init__(self, attacks_iter: int = 100, attack_restarts: int = 20, attack_epsilon: float = 2.0/255.0):
        self.attacks_iter = attacks_iter
        self.attack_restarts = attack_restarts
        self.attack_epsilon = attack_epsilon

    def generate(self, reference_model, test_model, img_tensor):

        attack = self.pgd_attack(reference_model, test_model, img_tensor)

        adv = self.compute_perturbed_input(img_tensor, attack)
        adv = torch.clamp(adv, 0, 1)  # Ensure the adversarial example is within valid range

        orig_class = torch.argmax(reference_model(adv), dim=1)

        adv_class = torch.argmax(test_model(adv), dim=1)

        return adv.detach(), orig_class, adv_class
    
    def init_delta(self, X):
        epsilon = self.attack_epsilon
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        # Delta should be in the range [0-X, 1-X] to ensure that X+delta is in the range [0,1]
        delta.data = torch.clamp(delta, 0-X, 1-X)
        return delta
    
    def compute_perturbed_input(self, X, delta):
        return X + delta
    
    def update_delta(self, X, delta, grad):
        epsilon = self.attack_epsilon
        attack_alpha = 2.5 * epsilon / self.attacks_iter
        d_a = torch.clamp(delta - attack_alpha * torch.sign(grad), -epsilon, epsilon)
        d_a = torch.clamp(d_a, 0-X, 1-X)
        return d_a

    def pgd_attack(self, reference_model, test_model, X):
        # Claude suggestion from *somewhere*??? -> Maybe RobustBench/AutoAttack?
        delta = self.init_delta(X)

        for _ in range(self.attack_restarts):
            cur_in = self.compute_perturbed_input(X, delta)
            reference_output = reference_model(cur_in)
            test_output = test_model(cur_in)
            is_same = (torch.argmax(reference_output, dim=1) == torch.argmax(test_output, dim=1))
            # We consider anything that is nan as successful attack so nan -> !is_same
            is_same = is_same & ~torch.isnan(reference_output).any(dim=1) & ~torch.isnan(test_output).any(dim=1)
            # print(f"Attack restart - Number of same predictions: {is_same.sum().item()}/{X.size(0)}")
            if not is_same.any():
                break
            # Reset delta with random noise for samples where the models have same prediction
            with torch.no_grad():
                noise = self.init_delta(X[is_same])
                delta.data[is_same] = noise
            
            active = is_same.nonzero(as_tuple=True)[0]

            # print("-------")
            for i in range(self.attacks_iter):
                if active.numel() == 0:
                    # print("active: All samples successfully attacked, breaking out of attack loop.")
                    break
                delta.grad = None
                delta.requires_grad = True

                X_a = X.index_select(0, active)
                delta_a = delta.index_select(0, active)

                reference_output = reference_model(self.compute_perturbed_input(X_a, delta_a))
                test_output = test_model(self.compute_perturbed_input(X_a, delta_a))

                reference_max = torch.argmax(reference_output, dim=1)
                test_max = torch.argmax(test_output, dim=1)

                # Selector: Models with same prediction
                with torch.no_grad():
                    is_same = (reference_max == test_max)
                    is_same = is_same & ~torch.isnan(reference_output).any(dim=1) & ~torch.isnan(test_output).any(dim=1)
                    # We only optimize for the samples where the models have the same prediction
                    if not is_same.any():
                        # print("same: All samples successfully attacked, breaking out of attack loop.")
                        break

                reference_2nd_largest = torch.topk(reference_output[is_same], 2).indices[:, 1]
                test_2nd_largest = torch.topk(test_output[is_same], 2).indices[:, 1]

                # Maximize confidence in reference model
                reference_loss = (reference_output[is_same].gather(1, reference_max[is_same].unsqueeze(1)) - reference_output[is_same].gather(1, reference_2nd_largest.unsqueeze(1)))
                # Minimize confidence in test model
                test_loss = (test_output[is_same].gather(1, test_max[is_same].unsqueeze(1)) - test_output[is_same].gather(1, test_2nd_largest.unsqueeze(1)))
                loss = (reference_loss + test_loss).mean()
                # print("Loss: ", loss)
                loss.backward()
                with torch.no_grad():
                    grad_a = delta.grad.index_select(0, active)
                    d_a = self.update_delta(X_a, delta_a, grad_a)
                    update_idx = active[is_same]
                    delta[update_idx] = d_a[is_same]
                active = active[is_same]
                # flush print statements
                # print(f"Attack iter {i} - Number of same predictions: {is_same.sum().item()}/{active.numel()}", flush=True)

        return delta

def get_rot_mat(theta):
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    row1 = torch.stack((cos_theta, -sin_theta, zeros), dim=-1)
    row2 = torch.stack((sin_theta, cos_theta, zeros), dim=-1)
    return torch.stack((row1, row2), dim=1)


def rot_img(x, theta, dtype):
    if theta.ndim == 0:
        theta = theta.unsqueeze(0)
    rot_mat = get_rot_mat(theta).to(device=x.device, dtype=dtype)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)

class RotationAdversary(Adversary):
    def init_delta(self, X):
        epsilon = self.attack_epsilon
        # 1 rotation angle per image in the batch, in the range [-epsilon, epsilon]
        delta = torch.zeros(X.size(0), device=X.device).uniform_(-epsilon, epsilon)
        return delta
    
    def compute_perturbed_input(self, X, delta):
        # Rotate each image by the corresponding angle in delta (batch-wise, differentiable)
        rot_mat = get_rot_mat(delta).to(device=X.device, dtype=X.dtype)
        grid = F.affine_grid(rot_mat, X.size(), align_corners=False)
        return F.grid_sample(X, grid, align_corners=False)

    
    def update_delta(self, X, delta, grad):
        epsilon = self.attack_epsilon
        attack_alpha = 2.5 * epsilon / self.attacks_iter
        d_a = torch.clamp(delta - attack_alpha * torch.sign(grad), -epsilon, epsilon)
        return d_a


class AffineAdversary(Adversary):
    """
    Adversary that perturbs images with a combination of rotation, x-shift and y-shift.
    delta has shape (batch, 3): [theta, tx, ty]
      - theta: rotation angle in radians, bounded by epsilon_rot
      - tx:    horizontal translation (fraction of image width), bounded by epsilon_tx
      - ty:    vertical translation (fraction of image height), bounded by epsilon_ty
    All three dimensions are differentiable via F.affine_grid / F.grid_sample.
    """

    def __init__(self, attacks_iter: int = 100, attack_restarts: int = 20,
                 epsilon_rot: float = 0.1, epsilon_tx: float = 0.05, epsilon_ty: float = 0.05):
        # Store individual bounds; use epsilon_rot as the generic attack_epsilon (unused directly)
        super().__init__(attacks_iter=attacks_iter, attack_restarts=attack_restarts,
                         attack_epsilon=epsilon_rot)
        self.epsilon_rot = epsilon_rot
        self.epsilon_tx = epsilon_tx
        self.epsilon_ty = epsilon_ty

    def _bounds(self, device):
        """Return lower and upper bound tensors of shape (3,) for [theta, tx, ty]."""
        lo = torch.tensor([-self.epsilon_rot, -self.epsilon_tx, -self.epsilon_ty], device=device)
        hi = torch.tensor([ self.epsilon_rot,  self.epsilon_tx,  self.epsilon_ty], device=device)
        return lo, hi

    def init_delta(self, X):
        lo, hi = self._bounds(X.device)
        # Uniform random initialisation within bounds, shape (batch, 3)
        delta = torch.zeros(X.size(0), 3, device=X.device).uniform_(0, 1)
        delta = lo + delta * (hi - lo)
        return delta

    def _build_affine_mat(self, delta):
        """Build (batch, 2, 3) affine matrix from delta = [theta, tx, ty]."""
        theta = delta[:, 0]
        tx    = delta[:, 1]
        ty    = delta[:, 2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # Row-major: [[cos, -sin, tx], [sin, cos, ty]]
        row1 = torch.stack([cos_t, -sin_t, tx], dim=-1)
        row2 = torch.stack([sin_t,  cos_t, ty], dim=-1)
        return torch.stack([row1, row2], dim=1)

    def compute_perturbed_input(self, X, delta):
        mat  = self._build_affine_mat(delta).to(device=X.device, dtype=X.dtype)
        grid = F.affine_grid(mat, X.size(), align_corners=False)
        return F.grid_sample(X, grid, align_corners=False)

    def update_delta(self, X, delta, grad):
        lo, hi = self._bounds(delta.device)
        attack_alpha = 2.5 / self.attacks_iter * (hi - lo) / 2.0  # scale step per dimension
        d_a = delta - attack_alpha * torch.sign(grad)
        d_a = torch.max(torch.min(d_a, hi), lo)
        return d_a

class VAEAdversary(Adversary):
    """
    Input images are encoded into a latent space by a pretrained VAE encoder.
    The adversary optimizes perturbations in the latent space, which are then decoded back into pixel space by the VAE decoder.
    """
    def __init__(self, encoder, decoder, attacks_iter: int = 100, attack_restarts: int = 20, attack_epsilon: float = 0.1):
        super().__init__(attacks_iter=attacks_iter, attack_restarts=attack_restarts, attack_epsilon=attack_epsilon)
        self.encoder = encoder.eval()  # Pretrained VAE encoder
        self.decoder = decoder.train()  # Pretrained VAE decoder (needed for gradients)
    
    def init_delta(self, X):
        # Encode input images to latent space
        with torch.no_grad():
            mu, logvar = self.encoder(X)
            z = mu # We use the mean as the latent representation for simplicity; could also sample with reparameterization trick if desired
        # Initialize delta in latent space
        epsilon = self.attack_epsilon
        delta = torch.zeros_like(z).uniform_(-epsilon, epsilon)
        return delta
    
    def compute_perturbed_input(self, X, delta):
        # Decode perturbed latent representation back to pixel space
        perturbed_z = self.encoder(X)[0] + delta  # Add delta to the mean latent representation
        adv = self.decoder(perturbed_z)
        return adv

    def update_delta(self, X, delta, grad):
        epsilon = self.attack_epsilon
        attack_alpha = 2.5 * epsilon / self.attacks_iter
        d_a = torch.clamp(delta - attack_alpha * torch.sign(grad), -epsilon, epsilon)
        return d_a

def robust_eval(reference_model, test_model, loader, criterion, device, adv):
    reference_model.eval()
    test_model.eval()
    normal_loss_ref, normal_loss_test, adv_loss_ref, adv_loss_test = 0.0, 0.0, 0.0, 0.0
    normal_acc_ref, normal_acc_test, adv_acc_ref, adv_acc_test = 0, 0, 0, 0
    normal_eq, adv_eq = 0, 0
    normal_nans_ref, normal_nans_test, adv_nans_ref, adv_nans_test = 0, 0, 0, 0
    total = 0
    adv_examples = []

    for test_input, test_label in loader:
        test_input, test_label = test_input.to(device), test_label.to(device)

        total += test_input.size(0)

        # Normal evaluation
        reference_outputs = reference_model(test_input)
        test_outputs = test_model(test_input)
        normal_loss_ref += criterion(reference_outputs, test_label).item()
        normal_loss_test += criterion(test_outputs, test_label).item()
        label_count = reference_outputs.argmax(1) == test_label
        label_count = label_count & ~torch.isnan(reference_outputs).any(dim=1)
        normal_acc_ref += (label_count).sum().item()
        label_count = test_outputs.argmax(1) == test_label
        label_count = label_count & ~torch.isnan(test_outputs).any(dim=1)
        normal_acc_test += (label_count).sum().item()
        label_count = reference_outputs.argmax(1) == test_outputs.argmax(1)
        label_count = label_count & ~torch.isnan(reference_outputs).any(dim=1) & ~torch.isnan(test_outputs).any(dim=1)
        normal_eq += (label_count).sum().item()
        # Count instances of nans in outputs (per image)
        normal_nans_ref += torch.isnan(reference_outputs).any(dim=1).sum().item()
        normal_nans_test += torch.isnan(test_outputs).any(dim=1).sum().item()

        # Adversarial evaluation
        adv_example, orig_class, adv_class = adv.generate(reference_model, test_model, test_input)
        reference_outputs_adv = reference_model(adv_example)
        test_outputs_adv = test_model(adv_example)
        adv_loss_ref += criterion(reference_outputs_adv, test_label).item()
        adv_loss_test += criterion(test_outputs_adv, test_label).item()
        label_count = reference_outputs_adv.argmax(1) == test_label
        label_count = label_count & ~torch.isnan(reference_outputs_adv).any(dim=1)
        adv_acc_ref += (label_count).sum().item()
        label_count = test_outputs_adv.argmax(1) == test_label
        label_count = label_count & ~torch.isnan(test_outputs_adv).any(dim=1)
        adv_acc_test += (label_count).sum().item()
        label_count = reference_outputs_adv.argmax(1) == test_outputs_adv.argmax(1)
        label_count = label_count & ~torch.isnan(reference_outputs_adv).any(dim=1) & ~torch.isnan(test_outputs_adv).any(dim=1)
        adv_eq += (label_count).sum().item()
        adv_examples.append((adv_example.cpu(), orig_class.cpu(), adv_class.cpu()))
        # Count instances of nans in outputs (per image)
        adv_nans_ref += torch.isnan(reference_outputs_adv).any(dim=1).sum().item()
        adv_nans_test += torch.isnan(test_outputs_adv).any(dim=1).sum().item()

    normal_loss_ref /= total
    normal_loss_test /= total
    adv_loss_ref /= total
    adv_loss_test /= total
    normal_acc_ref /= total
    normal_acc_test /= total
    adv_acc_ref /= total
    adv_acc_test /= total
    normal_eq /= total
    adv_eq /= total
    normal_nans_ref /= total
    normal_nans_test /= total
    adv_nans_ref /= total
    adv_nans_test /= total

    return {
        "normal_loss_ref": normal_loss_ref,
        "normal_loss_test": normal_loss_test,
        "adv_loss_ref": adv_loss_ref,
        "adv_loss_test": adv_loss_test,
        "normal_acc_ref": normal_acc_ref,
        "normal_acc_test": normal_acc_test,
        "adv_acc_ref": adv_acc_ref,
        "adv_acc_test": adv_acc_test,
        "normal_eq": normal_eq,
        "adv_eq": adv_eq,
        "normal_nans_ref": normal_nans_ref,
        "normal_nans_test": normal_nans_test,
        "adv_nans_ref": adv_nans_ref,
        "adv_nans_test": adv_nans_test,
        "total": total,
        "adv_examples": adv_examples
    }