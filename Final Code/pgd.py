import torch
import torch.nn as nn
import torch.nn.functional as F

threshold = 0.5  
hidden_size = 128
output_size = 1 
input_size = 299 * 299 * 3  
discriminator = Discriminator(input_size, hidden_size, output_size)

# Define the PGD defense function
def projected_gradient_descent(model, images, epsilon, alpha, num_iter):
    """
    Applies projected gradient descent (PGD) defense to images.

    Args:
        model: The model to be defended.
        images: Input images to defend.
        epsilon: Maximum perturbation allowed.
        alpha: Step size for each iteration.
        num_iter: Number of iterations for PGD.

    Returns:
        Defended images.
    """
    images = images.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        outputs = model(images)
        loss = F.binary_cross_entropy_with_logits(outputs, torch.ones_like(outputs))  # Binary cross-entropy loss
        loss.backward()

        # PGD step
        with torch.no_grad():
            images_grad = images.grad.sign()
            images = images + alpha * images_grad
            images = torch.max(torch.min(images, images + epsilon), images - epsilon)
            images = torch.clamp(images, 0, 1)
            images = images.detach().requires_grad_(True)

    return images


# Define PGD defense parameters
epsilon = 0.03 
alpha = 0.01   
num_iter = 10  

# Iterate through the dataset
for i, (batch, labels) in enumerate(train_loader):
    original_features = encoder(weight, batch)
    gaussian_noise = torch.randn(original_features.size(0), noise_size)
    attack_features = aacoder(original_features, gaussian_noise, aacoder_weight)
    restored_perturbations = decoder(original_features, attack_features)
    restored_perturbations_reshaped = restored_perturbations.view(restored_perturbations.size(0), -1, 1, 1)
    num_channels_original = batch.size(1)
    num_channels_perturbations = restored_perturbations_reshaped.size(1)
    if num_channels_original != num_channels_perturbations:
        restored_perturbations_reshaped = restored_perturbations_reshaped[:, :num_channels_original, :, :]
    adversarial_samples = batch + restored_perturbations_reshaped
    # Apply PGD defense to adversarial samples
    defended_samples = projected_gradient_descent(discriminator, adversarial_samples, epsilon, alpha, num_iter)
    flattened_defended_samples = defended_samples.view(defended_samples.size(0), -1)
    discriminator_output = discriminator(flattened_defended_samples)
    w_distance = wasserstein_distance(original_features, discriminator_output)
    classified_sample = "real" if w_distance >= threshold else "fake"

    # Print the classification result for each adversarial sample
    for j in range(len(batch)):
        print(f"Sample {i+1}, Adversarial Sample {j+1}: {classified_sample}")
        break
    break

