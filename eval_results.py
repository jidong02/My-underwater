import os
import glob
import argparse
import cv2
import core.metrics as Metrics

def imread_uint(path, n_channels=3):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if n_channels == 3:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def evaluate_results(result_dir):
    sr_paths = sorted([
        p for p in glob.glob(os.path.join(result_dir, "*_sr.png"))
        if not p.endswith("_sr_process.png")
    ])

    if len(sr_paths) == 0:
        raise RuntimeError(f"No *_sr.png found in {result_dir}")

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for sr_path in sr_paths:
        hr_path = sr_path.replace("_sr.png", "_hr.png")
        if not os.path.exists(hr_path):
            print(f"[Skip] HR not found for {os.path.basename(sr_path)}")
            continue

        sr_img = imread_uint(sr_path, n_channels=3)
        hr_img = imread_uint(hr_path, n_channels=3)

        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        print(f"{os.path.basename(sr_path):20s}  PSNR: {psnr:.4f}  SSIM: {ssim:.4f}")

    if count == 0:
        raise RuntimeError("No valid SR/HR pairs found.")

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print("\n====================")
    print(f"Result dir : {result_dir}")
    print(f"Num images : {count}")
    print(f"Avg PSNR   : {avg_psnr:.4f}")
    print(f"Avg SSIM   : {avg_ssim:.4f}")
    print("====================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Path to results folder")
    args = parser.parse_args()

    evaluate_results(args.result_dir)