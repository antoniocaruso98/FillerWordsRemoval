from main import myDataset
import os



def main():
    root_folder = os.path.join("..", "DATASET_COMPLETO_V2")
    # Read the training dataset to get the class order
    train_set = myDataset(root_folder, "train")
    class_order = train_set.classes_list  # Save the class order from the training set
    class_counts = train_set.labels_df.sort_values(by="label")["label"].value_counts(sort=False)
    print(f"Class order (from training set): {class_order}")
    print(f'#occorrenze: {class_counts}\n')

    # Apply the same class order to validation and test datasets
    test_set = myDataset(root_folder, "test")
    class_order = test_set.classes_list  # Save the class order from the training set
    class_counts = test_set.labels_df.sort_values(by="label")["label"].value_counts(sort=False)
    print(f"Class order (from test set): {class_order}")
    print(f'#occorrenze: {class_counts}\n')
    validation_set = myDataset(root_folder, "validation")
    class_order = validation_set.classes_list  # Save the class order from the training set
    class_counts = validation_set.labels_df.sort_values(by="label")["label"].value_counts(sort=False)
    print(f"Class order (from validation set): {class_order}")
    print(f'#occorrenze: {class_counts}\n')



if __name__ == "__main__":
    main()