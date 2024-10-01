

sft_prompt = "{instruction}"



all_prompt = {}

# =====================================================
# Task 1 -- Sequential Recommendation -- 17 Prompt
# =====================================================

seqrec_prompt = []

all_prompt["seqrec"] = seqrec_prompt



# ========================================================
# Task 2 -- Item2Index -- 19 Prompt
# ========================================================
# Remove periods when inputting

item2index_prompt = []


# ========================================================
# Description2Index

#####——6
prompt = {}
prompt["instruction"] = "Give the item index: {description}"
prompt["response"] = "{item}"
item2index_prompt.append(prompt)



all_prompt["item2index"] = item2index_prompt


# ========================================================
# Task 3 -- Index2Item --17 Prompt
# ========================================================
# Remove periods when inputting

index2item_prompt = []

# ========================================================
# Index2Description

#####——6
prompt = {}
prompt["instruction"] = "Give the description of item: {item}."
prompt["response"] = "{description}"
index2item_prompt.append(prompt)


all_prompt["index2item"] = index2item_prompt


# ========================================================
# Task 5 -- Product Search -- Prompt
# ========================================================


productsearch_prompt = []

#####——6
prompt = {}
prompt["instruction"] = "Find the item index: {query}"
prompt["response"] = "{item}"
productsearch_prompt.append(prompt)

all_prompt["productsearch"] = productsearch_prompt
