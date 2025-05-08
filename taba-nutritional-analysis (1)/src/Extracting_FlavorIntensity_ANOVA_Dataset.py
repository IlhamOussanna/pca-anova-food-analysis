#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
# For parsing stringified Python lists
import ast  


# In[17]:


# Load the Dataset
df = pd.read_csv("RAW_recipes.csv")


# In[18]:


# Extracting Cooking Method
def detect_cooking_method(step_text):
    try:
        text = ' '.join(ast.literal_eval(step_text)).lower()
    except:
        return None
    
    if "grill" in text:
        return "Grilled"
    elif "fry" in text:
        return "Fried"
    elif "bake" in text:
        return "Baked"
    elif "boil" in text:
        return "Boiled"
    elif "steam" in text:
        return "Steamed"
    else:
        return None

df["Cooking_Method"] = df["steps"].apply(detect_cooking_method)


# In[19]:


# Extracting Sauce Type and Flavor
def detect_sauce_type(tags, ingredients):
    try:
        tags_text = ' '.join(ast.literal_eval(tags)).lower()
    except:
        tags_text = ''
        
    try:
        ingredients_text = ' '.join(ast.literal_eval(ingredients)).lower()
    except:
        ingredients_text = ''
    
    text = tags_text + ' ' + ingredients_text
    
    if "spicy" in text:
        return "Spicy"
    elif "sweet" in text:
        return "Sweet"
    elif "savory" in text:
        return "Savory"
    elif "sour" in text:
        return "Sour"
    else:
        return None

df["Sauce_Type"] = df.apply(lambda row: detect_sauce_type(row["tags"], row["ingredients"]), axis=1)


# In[20]:


# Filtering Clean Rows
df_cleaned = df.dropna(subset=["Cooking_Method", "Sauce_Type"]).copy()


# In[21]:


# Adding Flavor Intensity Score for two-way ANOVA
np.random.seed(42)  # For reproducibility
df_cleaned["Flavor_Intensity_Score"] = np.random.randint(1, 11, size=len(df_cleaned))


# In[22]:


# Final Dataset for Analysis
df_final = df_cleaned[["name", "Cooking_Method", "Sauce_Type", "Flavor_Intensity_Score"]].reset_index(drop=True)
df_final.head()


# In[23]:


# Group by the two categorical factors
grouped = df_final.groupby(["Cooking_Method", "Sauce_Type"])

# Sample 3 rows per group to get a balanced design (3 observations per group)
df_balanced = grouped.apply(lambda x: x.sample(n=3, random_state=42)).reset_index(drop=True)


# In[24]:


# Should show 27 rows if 3x3 is achieved
print(df_balanced.shape)

# View the top few rows
df_balanced.head()


# In[25]:


df_balanced = df_balanced.drop(columns=["name"])


# In[14]:


df_balanced.to_csv("FlavorIntensity_ANOVA_Dataset.csv", index=False)


# In[ ]:




