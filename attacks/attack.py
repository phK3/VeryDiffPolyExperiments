import torch

class Adversary:
    
    def generate(self, model, img_array, epsilon: float, attack_iters: int = 100, attack_alpha: float = 1e-2, attack_restarts: int = 20):
        

        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.view(1, 3, 32, 32)

        prediction = model(img_tensor)
        y = torch.argmax(prediction[0], dim=0)

        attack_eps = epsilon

        attack = self.pgd_attack(model, img_tensor, y, attack_eps, attack_alpha, attack_iters, attack_restarts)

        adv = attack+img_tensor
        adv = torch.clamp(adv, 0, 1)  # Ensure the adversarial example is within valid range

        adv_class = torch.argmax(model(adv)[0], dim=0)

        return adv.detach(), y.item(), adv_class.item()

    def pgd_attack(self, model, X, y, epsilon, alpha, iters, restarts):
        max_delta = None

        for _ in range(restarts):
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
            delta.data = torch.clamp(delta, 0-X, 1-X)
            delta.requires_grad = True

            found_adv=False

            for _ in range(iters):
                output = model(X + delta)
                output_class = torch.argmax(output[0], dim=0)

                if output_class != y:
                    found_adv = True
                    break
                # Compute difference between output[y] and second largest output
                second_largest = torch.topk(output[0], 2)[0][1]
                loss = -(second_largest - output[0][y])
                
                loss.backward()
                grad = delta.grad.detach()
                d = torch.clamp(delta - alpha * torch.sign(grad), -epsilon, epsilon)
                d = torch.clamp(d, 0-X, 1-X)
                delta.data = d
                delta.grad.zero_()
            
            if found_adv:
                if max_delta is not None and torch.abs(delta).sum() < torch.abs(max_delta).sum():
                    max_delta = delta.detach()
                    output = model(X + delta)
                if max_delta is None:
                    output = model(X + delta)
                    max_delta = delta.detach()
        if max_delta is None:
            max_delta = delta.detach()

        return max_delta