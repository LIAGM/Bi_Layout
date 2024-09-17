import os

# BAD_PREDICTIONS_PATH = "/media/Pluto/frank/layout_ambiguity/mp3d/split"
# word_num = 2

BAD_PREDICTIONS_PATH = "/media/Pluto/frank/layout_ambiguity/zind"
word_num = 3

LED_SUBSET = os.path.join(BAD_PREDICTIONS_PATH, "led_bad_predictions.txt")
LGT_SUBSET = os.path.join(BAD_PREDICTIONS_PATH, "lgt_bad_predictions.txt")
DOP_SUBSET = os.path.join(BAD_PREDICTIONS_PATH, "dop_bad_predictions.txt")
FINAL_SUBSET = os.path.join(BAD_PREDICTIONS_PATH, "final_subset.txt")

# Read the contents of each file into a list
file1_contents = []
with open(LED_SUBSET, 'r') as file1:
  # {scene_id}_{image_id} for mp3d, {scene_id}_pano_{image_id} for zind
  file1_contents = ['_'.join(x.rstrip().split('_')[:word_num]) for x in file1]

file2_contents = []
with open(LGT_SUBSET , 'r') as file2:
  file2_contents = ['_'.join(x.rstrip().split('_')[:word_num]) for x in file2]

file3_contents = []
with open(DOP_SUBSET, 'r') as file3:
  file3_contents = ['_'.join(x.rstrip().split('_')[:word_num]) for x in file3]

# Create a set of all of the names
union_contents = set(file1_contents + file2_contents + file3_contents)

# Write the union of the names to a new file
with open(FINAL_SUBSET, 'w') as outfile:
  for content in union_contents:
    outfile.write(content + '\n')