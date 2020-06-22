basic:
	python generate_dataset_with_one_type.py \
	--n_images 10
	#DONE

clean:
	rm -r out
	rm out.txt
	#DONE

dataset1:
	python generate_dataset_with_one_type.py \
	--n_images 10 \
	--out_file dataset1/train.txt \
	--class_id 3 \
	--out_folder_with_selected_images dataset1/samples

dataset2:
	time python generate_dataset_with_one_type.py \
	--n_images 50 \
	--out_file dataset2/train.txt \
	--class_id 3 \
	--out_folder_with_selected_images dataset2/samples

dataset3: # it generated only 573 images due to configuration - I'll fix it
	time python generate_dataset_with_one_type.py \
	--n_images 2000 \
	--out_file dataset3/train.txt \
	--class_id 3 \
	--out_folder_with_selected_images dataset3/samples

dataset4:
	time --format="took %E" python generate_dataset_with_one_type.py \
	--n_images 2000 \
	--out_file dataset4/train.txt \
	--class_id 4 \
	--out_folder_with_selected_images dataset4/samples

balanced_dataset:
	time --format="took %E" \
	python generate_dataset_n_elements.py \
	--n_cpu 8 \
	--iter 100 \
	--epochs 10000 \
	--n_classes 14 \
	--expected_capacity 500 \
	--use_absolute_path 1 \
	--load_collection out.txt
