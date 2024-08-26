import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import cv2
from lpips import LPIPS
import pandas as pd
from tqdm import tqdm
from pytorch_lightning.loggers import CometLogger
from dotmap import DotMap
from utils.generate_graphs import generate_graphs_test

from utils.metrics import compute_psnr, compute_ssim, compute_lpips, compute_vmaf
from data.synthetic_video_dataset import SyntheticVideoDataset
from utils.utils import PROJECT_ROOT, device
import json

def test(args: DotMap, net: nn.Module, logger: CometLogger):
    """
    Test the model on the synthetic dataset test split.

    Args:
        args: Arguments
        net: Network to be tested
        logger: Comet ML logger
    """
    data_base_path = Path(args.data_base_path)
    test_path = data_base_path / "test"
    gt_videos_path = test_path / "gt" / "videos"
    input_videos_path = test_path / "input" / "videos"
    results_path = PROJECT_ROOT / "experiments" / args.experiment_name / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    videos_path = results_path / Path("videos")
    restored_videos_path = videos_path / "restored"
    restored_videos_path.mkdir(parents=True, exist_ok=True)
    combined_videos_path = videos_path / "combined"
    combined_videos_path.mkdir(parents=True, exist_ok=True)

    data_log_path = PROJECT_ROOT / "experiments" / args.experiment_name / "data_log.json"

    with open(str(data_log_path), "r+") as json_file:
            data = json.load(json_file)
            data["test"] = {}
            json_file.seek(0)
            json.dump(data, json_file, indent=4)
            json_file.truncate()

    test_dataset = SyntheticVideoDataset(data_base_path=test_path,
                                         num_neighbour_frames=args.num_neighbour_frames,
                                         preprocess_mode="none")

    video_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }

    input_video_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }

    net.eval()
    net.to(device)

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    lpips_net = LPIPS(pretrained=True, net='alex').to(device)
    count_videos = 0
    count_frames = 0
    if not args.no_vmaf:
        column_names = ["Name", "PSNR", "SSIM", "LPIPS", "VMAF"]
        metrics_list = ["PSNR", "SSIM", "LPIPS", "VMAF"]
    else:
        column_names = ["Name", "PSNR", "SSIM", "LPIPS"]
        metrics_list = ["PSNR", "SSIM", "LPIPS"]
    single_metrics_results = {}
    total_metrics_results = {metric: 0 for metric in metrics_list}
    output_csv = [[]]
    output_csv_path = results_path / "metrics_results.csv"

    single_metrics_results_input = {}
    total_metrics_results_input = {metric: 0 for metric in metrics_list}
    output_csv_input = [[]]
    output_csv_path_input = results_path / "metrics_results_input.csv"

    last_clip = ""
    restored_video_writer = cv2.VideoWriter()
    combined_video_writer = cv2.VideoWriter()
    combined_resized_video_writer = cv2.VideoWriter()

    for batch in tqdm(dataloader):
        img_name = batch["img_name"]
        imgs_lq = batch["imgs_lq"].to(device, non_blocking=True)
        img_gt = batch["img_gt"]

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = torch.clamp(net(imgs_lq), 0, 1).permute(0, 2, 3, 1).cpu().numpy()

        """
        imgs_lq with shape (b, t, c, h, w)
        img_gt with shape (b, c, h, w)
        output with shape (b, c, h, w)
        """

        for i in range(output.shape[0]):
            video_clip = Path(img_name[i]).parent
            (results_path / Path(video_clip)).mkdir(parents=True, exist_ok=True)

            restored = (output[i] * 255).astype(np.uint8)
            restored = restored[..., ::-1]    # RGB -> BGR

            input = imgs_lq[i, args.num_neighbour_frames].permute(1, 2, 0).cpu().numpy()
            input = (input * 255).astype(np.uint8)
            input = input[..., ::-1]  # RGB -> BGR

            gt = img_gt[i].permute(1, 2, 0).cpu().numpy()
            gt = (gt * 255).astype(np.uint8)
            gt = gt[..., ::-1]  # RGB -> BGR

            if video_clip != last_clip:
                restored_video_writer.release()
                combined_video_writer.release()
                combined_resized_video_writer.release()

                if last_clip != "":
                    gt_video_path = gt_videos_path / f"{last_clip}.mp4"
                    input_video_path = input_videos_path / f"{last_clip}.mp4"
                    if not args.no_vmaf:
                        single_metrics_results["VMAF"] = compute_vmaf(restored_video_path, gt_video_path,
                                                                    width=restored.shape[0], height=restored.shape[1])
                        single_metrics_results_input["VMAF"] = compute_vmaf(input_video_path, gt_video_path,
                                                                    width=restored.shape[0], height=restored.shape[1])
                    for metric in single_metrics_results.keys():
                        if metric != "VMAF":
                            single_metrics_results[metric] /= count_frames
                        total_metrics_results[metric] += single_metrics_results[metric]
                    output_csv_row = list(single_metrics_results.values())
                    output_csv_row.insert(0, last_clip)
                    output_csv.append(output_csv_row)

                    for metric in single_metrics_results_input.keys():
                        if metric != "VMAF":
                            single_metrics_results_input[metric] /= count_frames
                        total_metrics_results_input[metric] += single_metrics_results_input[metric]
                    output_csv_row_input = list(single_metrics_results_input.values())
                    output_csv_row_input.insert(0, last_clip)
                    output_csv_input.append(output_csv_row_input)


                    if data_log_path is not None:
                        with open(str(data_log_path), "r+") as json_file:
                            data = json.load(json_file)
                            data["test"][str(last_clip)] = {}
                            data["test"][str(last_clip)]["restored"] = video_metrics
                            data["test"][str(last_clip)]["input"] = input_video_metrics
                            json_file.seek(0)
                            json.dump(data, json_file, indent=4)
                            json_file.truncate()

                    for video_metric_name in video_metrics.keys():
                        video_metrics[video_metric_name] = []
                        input_video_metrics[video_metric_name] = []

                last_clip = video_clip
                restored_video_path = restored_videos_path / f"{last_clip}.mp4"
                combined_video_path = combined_videos_path / f"{last_clip}.mp4"
                combined_resized_video_path = combined_videos_path / f"{last_clip}_combined_resized.mp4"
                
                restored_shape = (restored.shape[1], restored.shape[0])
                restored_video_writer = cv2.VideoWriter(str(restored_video_path),
                                                        cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                                        restored_shape)

                combined_shape = (restored.shape[1] * 3, restored.shape[0])
                combined_video_writer = cv2.VideoWriter(str(combined_video_path),
                                                        cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                                        combined_shape)
                combined_resized_video_writer = cv2.VideoWriter(str(combined_resized_video_path),
                                                        cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                                        combined_shape)

                single_metrics_results = {metric: 0 for metric in metrics_list}

                single_metrics_results_input = {metric: 0 for metric in metrics_list}

                count_videos += 1
                count_frames = 0

            restored_video_writer.write(restored)

            input_big = np.zeros_like(restored)
            h, w, _ = input.shape
            h_big, w_big, _ = restored.shape
            h_start = int((h_big - h) // 2)
            w_start = int((w_big - w) // 2)
            input_big[h_start:h_start+h,w_start:w_start+w, :] = input

            combined = np.hstack((input_big, restored, gt))
            combined_video_writer.write(combined)

            input_resized = cv2.resize(input, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

            combined_resized = np.hstack((input_resized, restored, gt))
            combined_resized_video_writer.write(combined_resized)

            psnr = compute_psnr(restored, gt)
            ssim = compute_ssim(restored, gt)
            lpips = compute_lpips(restored, gt, lpips_net, device)

            video_metrics["psnr"].append(float(psnr))
            video_metrics["ssim"].append(float(ssim))
            video_metrics["lpips"].append(float(lpips))
            
            single_metrics_results["PSNR"] += psnr
            single_metrics_results["SSIM"] += ssim
            single_metrics_results["LPIPS"] += lpips


            psnr_input = compute_psnr(input_resized, gt)
            ssim_input = compute_ssim(input_resized, gt)
            lpips_input = compute_lpips(input_resized, gt, lpips_net, device)

            input_video_metrics["psnr"].append(float(psnr_input))
            input_video_metrics["ssim"].append(float(ssim_input))
            input_video_metrics["lpips"].append(float(lpips_input))

            single_metrics_results_input["PSNR"] += psnr_input
            single_metrics_results_input["SSIM"] += ssim_input
            single_metrics_results_input["LPIPS"] += lpips_input

            count_frames += 1

            cv2.imwrite(f"{results_path}/{img_name[i]}", restored)
            torch.cuda.empty_cache()

    restored_video_writer.release()
    combined_video_writer.release()
    combined_resized_video_writer.release()

    # Compute metrics of last video
    gt_video_path = gt_videos_path / f"{last_clip}.mp4"
    input_video_path = input_videos_path / f"{last_clip}.mp4"
    if not args.no_vmaf:
        single_metrics_results["VMAF"] = compute_vmaf(restored_video_path, gt_video_path,
                                                    width=restored.shape[0], height=restored.shape[1])
        single_metrics_results_input["VMAF"] = compute_vmaf(input_video_path, gt_video_path,
                                        width=restored.shape[0], height=restored.shape[1])
            
    for metric in single_metrics_results.keys():
        if metric != "VMAF":
            single_metrics_results[metric] /= count_frames
        total_metrics_results[metric] += single_metrics_results[metric]
    output_csv_row = list(single_metrics_results.values())
    output_csv_row.insert(0, last_clip)
    output_csv.append(output_csv_row)

    for metric in single_metrics_results_input.keys():
        if metric != "VMAF":
            single_metrics_results_input[metric] /= count_frames
        total_metrics_results_input[metric] += single_metrics_results_input[metric]
    output_csv_row_input = list(single_metrics_results_input.values())
    output_csv_row_input.insert(0, last_clip)
    output_csv_input.append(output_csv_row_input)

    if data_log_path is not None:
        with open(str(data_log_path), "r+") as json_file:
            data = json.load(json_file)
            data["test"][str(last_clip)] = {}
            data["test"][str(last_clip)]["restored"] = video_metrics
            data["test"][str(last_clip)]["input"] = input_video_metrics
            json_file.seek(0)
            json.dump(data, json_file, indent=4)
            json_file.truncate()

    for metric in total_metrics_results.keys():
        total_metrics_results[metric] /= count_videos
        logger.experiment.log_metric(metric, total_metrics_results[metric])
    output_csv_row = list(total_metrics_results.values())
    output_csv_row.insert(0, "Total")
    output_csv.append(output_csv_row)

    for metric in total_metrics_results_input.keys():
        total_metrics_results_input[metric] /= count_videos
    output_csv_row_input = list(total_metrics_results_input.values())
    output_csv_row_input.insert(0, "Total")
    output_csv_input.append(output_csv_row_input)

    df = pd.DataFrame(output_csv)
    df.columns = column_names
    df.to_csv(output_csv_path, index=False)

    df_input = pd.DataFrame(output_csv_input)
    df_input.columns = column_names
    df_input.to_csv(output_csv_path_input, index=False)

    if args.generate_graphs:
        generate_graphs_test(data_log_path)
