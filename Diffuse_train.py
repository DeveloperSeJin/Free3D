pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  
instance_data_dir="./data/"
output_dir="./models" 
class_data_dir=./real_reg/samples_cat/ 
with_prior_preservation --real_prior --prior_loss_weight=1.0 
class_prompt="" --num_class_images=200 
instance_prompt="photo of a chair"  
resolution=512  
train_batch_size=2  
learning_rate=1e-5  
lr_warmup_steps=0 
max_train_steps=250 
scale_lr --hflip  
modifier_token "<chairs>" 