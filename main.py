import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.ImagesDataset import ImagesDataset
from pivot_tuning import GeneratorTuning
from adversarial_optimization import Adversarial_Opt

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="CLIP2Protect")

    # arguments for the stage-1 (generator tuning)
    parser.add_argument('--data_dir', type=str, default='input_images', help='The directory of input images')
    parser.add_argument('--noise_path', type=str, default='noises.pt', help='Path to save the generator noise file')
    parser.add_argument('--inverted_image_path', type=str, default='inverted_images', help='Path to save the inverted images in the first stage')
    parser.add_argument('--latent_path', type=str, default='latents.pt', help='Path to the latent file calculated by e4e method')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_dir', help='Path to the finetuned generator weights in the first stage')
    parser.add_argument('--num_steps', type=int, default=150, help='The number of steps for generator tuning')
    parser.add_argument('--gt_lr', type=float, default=0.0005, help='Learning rate for generator tuning')

    # arguments for the stage-2 (adversarial optimization)
    parser.add_argument('--num_aug', type=int, default=1)
    parser.add_argument('--source_text', type=str, default='face')
    parser.add_argument('--makeup_prompt', type=str, default='red lipstick')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--target_choice', type=str, default='2', help='Choice of target identity, as in AMT-GAN. We use 4 target identities provided by AMT-GAN')
    parser.add_argument('--model', type=str, default='mobile_face', help = 'facenet','irse50','ir152')
    parser.add_argument('--impersonate', type=bool, default=True, help = 'For protection during impersonation')
    parser.add_argument('--noise_optimize', type=bool, default=True, help = 'Use noise vectors in StyleGAN during optimization')
    parser.add_argument('--margin', type=int, default=0, help = 'MTCNN margin')
    parser.add_argument('--lambda_lat', type=float, default=0.01)
    parser.add_argument('--lambda_clip', type=float, default=0.3)
    parser.add_argument('--lambda_adv', type=float, default=0.7)
    parser.add_argument('--protected_face_dir', type=str, default='results')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Define your dataset
    dataset = ImagesDataset(args.data_dir, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    # Create the DataLoader
    args.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create an instance of the the stage-1 (generator tuning)
    generator_tuning = GeneratorTuning(args)
    generator_tuning.run()

    # Create an instance of the the stage-2 (adversarial optimization)
    adversarial_opt = Adversarial_Opt(args)
    adversarial_opt.run()
