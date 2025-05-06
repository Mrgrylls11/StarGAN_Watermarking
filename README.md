python main.py --mode train --dataset CelebA \
               --image_size 128 --batch_size 1 \
               --num_iters 10000 --model_save_step 2000 --sample_step 500 \
               --log_step 10 --num_workers 2 \
               --apply_watermark True \
               --watermark_path watermark/watermarks/logo.png \
               --watermark_size_percent 10 \
               --watermark_position bottom-right \
               --watermark_output_dir stargan_results/watermarked_results \
               --save_noisy_intermediate True \
               --watermark_intermediate_dir stargan_results/intermediate_noisy
               # Add other parameters like --selected_attrs if needed
