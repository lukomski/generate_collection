all:
generatate_dataset_n_elements:
	python generate_dataset_n_elements.py \
	--override_out_file True \
	--use_absolute_path True \
	--expected_capacity 100 \
	--out_file dataset_n_100.txt \
	--epochs 100 \
	--n_cpu 1 \
	--iter 50 \
	--load_collection dataset_n_100.txt

