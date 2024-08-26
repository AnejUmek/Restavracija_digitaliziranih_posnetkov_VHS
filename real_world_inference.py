import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import cv2
from tqdm import tqdm
import torchvision
import shutil

from data.real_world_video_dataset import RealWorldVideoDataset
from models.swin_unet import SwinUNet

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def real_world_test(args):
    """
    Restore a real-world video (i.e. without ground truth) using the pretrained model.
    """
    
    if args.input_path.is_dir():
        input_paths = sorted(args.input_path.glob("*.mp4"))
    else:
        input_paths = [args.input_path]

    for input_path in input_paths:
        input_video_name = input_path.stem
        output_folder = args.output_path / input_video_name
        output_folder.mkdir(parents=True, exist_ok=False)
        output_folder.mkdir(parents=True, exist_ok=True)
        input_frames_folder = output_folder / "input_frames"
        input_frames_folder.mkdir(parents=True, exist_ok=True)
        restored_frames_folder = output_folder / "restored_frames"
        restored_frames_folder.mkdir(parents=True, exist_ok=True)

        ### 1) Frames extraction
        print("Extracting frames from the video...")
        input_video = cv2.VideoCapture(str(input_path))
        fps = input_video.get(cv2.CAP_PROP_FPS)
        frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(frame_count)):
            success, frame = input_video.read()
            if not success:
                raise Exception("Failed to read frame from video")
            padded_i = str(i).zfill(len(str(frame_count)))      # Pad to a number of digits large enough to contain the total number of frames
            cv2.imwrite(str(input_frames_folder / f"{padded_i}.{args.frame_format}"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        input_video.release()

        ### 3) Video restoration
        print("Restoring the video...")
        dataset = RealWorldVideoDataset(input_frames_folder, num_neighbour_frames=args.num_neighbour_frames,
                                        preprocess_mode="none",
                                        patch_size=args.patch_size,
                                        frame_format=args.frame_format)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                shuffle=False, pin_memory=True, drop_last=False)

        new_frame_width = 2 * frame_width
        new_frame_height = 2 * frame_height

        output_video = cv2.VideoWriter(str(output_folder / f"restored_{input_video_name}.mp4"),
                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_frame_width, new_frame_height))
        if args.generate_combined_video:
            combined_output_video = cv2.VideoWriter(str(output_folder / f"combined_{input_video_name}.mp4"),
                                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_frame_width * 2, new_frame_height))
        else:
            combined_output_video = None

        # Load model
        model = SwinUNet()
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items() if k.startswith("net.")}
        model.load_state_dict(state_dict, strict=True)
        model = model.eval().to(device)

        for batch in tqdm(dataloader, desc="Restoring frames"):
            imgs_lq = batch["imgs_lq"]
            img_names = batch["img_name"]

            # Image size must be divisible by 16 (due to the 4 downsampling operations)
            pad_width = (16 - (frame_width % 16)) % 16
            pad_height = (16 - (frame_height % 16)) % 16
            pad = (0, pad_width, 0, pad_height)
            imgs_lq = F.pad(imgs_lq, pad=pad, mode="constant", value=0).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(imgs_lq)
                output = torch.clamp(output, min=0, max=1)

            for i, img_name in enumerate(img_names):
                img_num = int(img_name[:-4])
                restored_frame = output[i]
                restored_frame = torchvision.transforms.functional.crop(restored_frame, top=0, left=0, height=new_frame_height, width=new_frame_width)
                restored_frame = restored_frame.cpu().numpy().transpose(1, 2, 0) * 255
                restored_frame = restored_frame.astype(np.uint8)
                restored_frame = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(restored_frames_folder / f"{img_num}.{args.frame_format}"), restored_frame)

                # Reconstruct the video
                output_video.write(restored_frame)
                if args.generate_combined_video:

                    input_frame = imgs_lq[i, args.num_neighbour_frames]
                    input_frame = torchvision.transforms.functional.crop(input_frame, top=0, left=0, height=frame_height, width=frame_width)
                    input_frame = input_frame.cpu().numpy().transpose(1, 2, 0) * 255
                    input_frame = input_frame.astype(np.uint8)
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)

                    input_frame_big = np.zeros_like(restored_frame)
                    h_start = int((new_frame_height - frame_height) // 2)
                    w_start = int((new_frame_width - frame_width) // 2)
                    input_frame_big[h_start:(h_start+frame_height),w_start:(w_start+frame_width), :] = input_frame

                    combined_frame = np.concatenate((input_frame_big, restored_frame), axis=1)
                    combined_output_video.write(combined_frame)

        output_video.release()
        if args.generate_combined_video:
            combined_output_video.release()

        # Free memory
        del model
        del imgs_lq
        torch.cuda.empty_cache()

        if args.no_intermediate_products:
            print("Deleting intermediate products...")
            (output_folder / f"restored_{input_video_name}.mp4").rename(Path(args.output_path) / f"restored_{input_video_name}.mp4")
            if args.generate_combined_video:
                (output_folder / f"combined_{input_video_name}.mp4").rename(Path(args.output_path) / f"combined_{input_video_name}.mp4")
            shutil.rmtree(output_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to the video or a folder of videos to restore")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--checkpoint-path", type=str, default="experiments/pretrained_model/checkpoint.pth",
                        help="Path to the pretrained model checkpoint")
    parser.add_argument("--num-neighbour-frames", type=int, default=3,
                        help="Number of neighbour frames to each side")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Maximum patch size for --preprocess-mode ['crop', 'resize']")
    parser.add_argument("--frame-format", type=str, default="jpg",
                        help="Frame format of the extracted and restored frames")
    parser.add_argument("--generate-combined-video", action="store_true",
                        help="Whether to generate the combined video (i.e. input and restored videos side by side)")
    parser.add_argument("--no-intermediate-products", action="store_true",
                        help="Whether to delete intermediate products (i.e. input frames, restored frames, references)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers of the data loader")

    args = parser.parse_args()

    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)
    real_world_test(args)
