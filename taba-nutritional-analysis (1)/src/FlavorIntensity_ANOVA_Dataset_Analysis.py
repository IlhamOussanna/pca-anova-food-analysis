#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[6]:


# Loading the Dataset
df_anova = pd.read_csv("FlavorIntensity_ANOVA_Dataset.csv")


# In[13]:


# Running Two-Way ANOVA test
model = smf.ols('Flavor_Intensity_Score ~ C(Cooking_Method) + C(Sauce_Type) + C(Cooking_Method):C(Sauce_Type)', data=df_anova).fit()

# Displaying the ANOVA Table
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# In[10]:


# Main Effect Plot: Cooking Method
plt.figure(figsize=(8, 5))
sns.barplot(x="Cooking_Method", y="Flavor_Intensity_Score", data=df_anova, ci="sd", palette="Set2")
plt.title("Main Effect of Cooking Method on Flavor Intensity")
plt.ylabel("Mean Flavor Intensity Score")
plt.xlabel("Cooking Method")
plt.tight_layout()
plt.show()


# In[12]:


# Interaction Plot: Cooking Method × Sauce Type
plt.figure(figsize=(8, 5))
sns.pointplot(x="Cooking_Method", y="Flavor_Intensity_Score", hue="Sauce_Type",
              data=df_anova, dodge=True)
plt.title("Interaction Effect: Cooking Method × Sauce Type")
plt.ylabel("Mean Flavor Intensity Score")
plt.xlabel("Cooking Method")
plt.legend(title="Sauce Type")
plt.tight_layout()
plt.show()


# In[ ]:




