import os

root_folder = "clean_segments_v2\\validation"
output_file = open("validation_nonfiller_clips_labels_v2.csv", "w", encoding="utf-8")
for nome_file in os.listdir(root_folder):
    s = f"{nome_file},Nonfiller,0.0,0.0\n"
    output_file.write(s)
