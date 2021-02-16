import pandas as pd

hard_df = pd.read_csv("data/images/labels.csv")
soft_df = pd.read_csv("data/images/soft_labels.csv")

sorted_hard_df = hard_df.sort_values("image_id").reset_index(drop=True)
sorted_soft_df = soft_df.sort_values("image_id").reset_index(drop=True)

sorted_hard_df.to_csv("data/images/sorted_labels.csv", index=False)
sorted_soft_df.to_csv("data/images/sorted_soft_labels.csv", index=False)
