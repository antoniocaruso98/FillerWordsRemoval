import os

root_folder = "train_clips"
output_file = open("train_nonfiller_clips_labels.csv", "w")
for nome_file in os.listdir(root_folder):
    s = f"{nome_file},Nonfiller,0.0,0.0\n"
    output_file.write(s)
