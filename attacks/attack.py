import torch

class Adversary:

    def __init__(self, attacks_iter: int = 100, attack_restarts: int = 20, attack_epsilon: float = 2.0/255.0):
        self.attacks_iter = attacks_iter
        self.attack_restarts = attack_restarts
        self.attack_epsilon = attack_epsilon

    def generate(self, reference_model, test_model, img_tensor):

        attack = self.pgd_attack(reference_model, test_model, img_tensor)

        adv = attack+img_tensor
        adv = torch.clamp(adv, 0, 1)  # Ensure the adversarial example is within valid range

        orig_class = torch.argmax(reference_model(adv), dim=1)

        adv_class = torch.argmax(test_model(adv), dim=1)

        return adv.detach(), orig_class, adv_class

    def pgd_attack(self, reference_model, test_model, X):
        # Claude suggestion from *somewhere*??? -> Maybe RobustBench/AutoAttack?
        epsilon = self.attack_epsilon
        attack_alpha = 2.5 * epsilon / self.attacks_iter

        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        # Delta should be in the range [0-X, 1-X] to ensure that X+delta is in the range [0,1]
        delta.data = torch.clamp(delta, 0-X, 1-X)
        # Needs to be differentiable for gradient ascent
        delta.requires_grad = True

        for _ in range(self.attack_restarts):
            reference_output = reference_model(X + delta)
            test_output = test_model(X + delta)
            is_same = (torch.argmax(reference_output, dim=1) == torch.argmax(test_output, dim=1))
            # print(f"Attack restart - Number of same predictions: {is_same.sum().item()}/{X.size(0)}")
            if not is_same.any():
                break
            # Reset delta with random noise for samples where the models have same prediction
            with torch.no_grad():
                noise = torch.zeros_like(delta[is_same]).uniform_(-epsilon, epsilon)
                delta.data[is_same] = torch.clamp(noise, 0-X[is_same], 1-X[is_same])
            
            active = is_same.nonzero(as_tuple=True)[0]

            for _ in range(self.attacks_iter):
                if active.numel() == 0:
                    break
                delta.grad = None

                X_a = X.index_select(0, active)
                delta_a = delta.index_select(0, active)

                reference_output = reference_model(X_a + delta_a)
                test_output = test_model(X_a + delta_a)

                reference_max = torch.argmax(reference_output, dim=1)
                test_max = torch.argmax(test_output, dim=1)

                # Selector: Models with same prediction
                with torch.no_grad():
                    is_same = (reference_max == test_max)
                    # We only optimize for the samples where the models have the same prediction
                    if not is_same.any():
                        break

                reference_2nd_largest = torch.topk(reference_output[is_same], 2).indices[:, 1]
                test_2nd_largest = torch.topk(test_output[is_same], 2).indices[:, 1]

                # Maximize confidence in reference model
                reference_loss = (reference_output[is_same].gather(1, reference_max[is_same].unsqueeze(1)) - reference_output[is_same].gather(1, reference_2nd_largest.unsqueeze(1)))
                # Minimize confidence in test model
                test_loss = (test_output[is_same].gather(1, test_max[is_same].unsqueeze(1)) - test_output[is_same].gather(1, test_2nd_largest.unsqueeze(1)))
                loss = (reference_loss + test_loss).mean()
                loss.backward()
                with torch.no_grad():
                    grad_a = delta.grad.index_select(0, active)
                    d_a = torch.clamp(delta_a - attack_alpha * torch.sign(grad_a), -epsilon, epsilon)
                    d_a = torch.clamp(d_a, 0-X_a, 1-X_a)
                    update_idx = active[is_same]
                    delta[update_idx] = d_a[is_same]
                active = active[is_same]

        return delta

def robust_eval(reference_model, test_model, loader, criterion, device, adv):
    reference_model.eval()
    test_model.eval()
    normal_loss_ref, normal_loss_test, adv_loss_ref, adv_loss_test = 0.0, 0.0, 0.0, 0.0
    normal_acc_ref, normal_acc_test, adv_acc_ref, adv_acc_test = 0, 0, 0, 0
    normal_eq, adv_eq = 0, 0
    total = 0
    for test_input, test_label in loader:
        test_input, test_label = test_input.to(device), test_label.to(device)

        total += test_input.size(0)

        # Normal evaluation
        reference_outputs = reference_model(test_input)
        test_outputs = test_model(test_input)
        normal_loss_ref += criterion(reference_outputs, test_label).item()
        normal_loss_test += criterion(test_outputs, test_label).item()
        normal_acc_ref += (reference_outputs.argmax(1) == test_label).sum().item()
        normal_acc_test += (test_outputs.argmax(1) == test_label).sum().item()
        normal_eq += (reference_outputs.argmax(1) == test_outputs.argmax(1)).sum().item()

        # Adversarial evaluation
        adv_example, orig_class, adv_class = adv.generate(reference_model, test_model, test_input)
        reference_outputs_adv = reference_model(adv_example)
        test_outputs_adv = test_model(adv_example)
        adv_loss_ref += criterion(reference_outputs_adv, test_label).item()
        adv_loss_test += criterion(test_outputs_adv, test_label).item()
        adv_acc_ref += (reference_outputs_adv.argmax(1) == test_label).sum().item()
        adv_acc_test += (test_outputs_adv.argmax(1) == test_label).sum().item()
        adv_eq += (reference_outputs_adv.argmax(1) == test_outputs_adv.argmax(1)).sum().item()

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
        "total": total
    }