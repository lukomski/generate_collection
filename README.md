# generate collection

Skrypt do generowania zbiorów N elementowych. Skrypt potrzebuje zbioru zdjęć, ich opisów oraz pliku z klasami. 
Zajmuje się znalezienie w zbiorze podzbioru o określonej wielkości dla każdej klasy

### Pomoc uruchomienia
```
usage: generate_dataset_n_elements.py [-h] [--n_cpu N_CPU]

                                      [--folder_with_files FOLDER_WITH_FILES]
                                      [--out_file OUT_FILE]
                                      [--override_out_file OVERRIDE_OUT_FILE]
                                      [--use_absolute_path USE_ABSOLUTE_PATH]
                                      [--class_path CLASS_PATH]
                                      [--expected_capacity EXPECTED_CAPACITY]
                                      [--epochs EPOCHS] [--iter ITER]
                                      [--load_collection LOAD_COLLECTION]

optional arguments:
  -h, --help            show this help message and exit
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --folder_with_files FOLDER_WITH_FILES
                        path to folder with images and labels
  --out_file OUT_FILE   file with paths selected by the program
  --override_out_file OVERRIDE_OUT_FILE
                        enable override output file
  --use_absolute_path USE_ABSOLUTE_PATH
                        use absolute paths in out data
  --class_path CLASS_PATH
                        path to class label file
  --expected_capacity EXPECTED_CAPACITY
                        expected capacity of each class probe
  --epochs EPOCHS       number of epochs
  --iter ITER           iterations in each epoch
  --load_collection LOAD_COLLECTION
                        path to saved collection to begin from
```
