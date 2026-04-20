from experiments import ABLATION_EXPERIMENTS
from test import test_model
from train import train_model


def main():
    for experiment_name in ABLATION_EXPERIMENTS[1:]:
        train_model(variant_name=experiment_name)

    results = [test_model(experiment_name) for experiment_name in ABLATION_EXPERIMENTS]

    print("\nAblation summary")
    for result in results:
        print(
            f"{result['experiment']}: "
            f"FID={result['fid']:.2f}, "
            f"LPIPS={result['lpips']:.4f}, "
            f"PSNR={result['psnr']:.2f}, "
            f"SSIM={result['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
