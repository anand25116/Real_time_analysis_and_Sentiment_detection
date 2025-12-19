import pandas as pd
import requests

df = pd.read_csv("amazon_sales.csv")

def is_valid_image(url):
    if pd.isna(url):
        return False
    url = str(url)
    if not url.startswith(("http://", "https://")):
        return False
    try:
        response = requests.get(url, timeout=2, stream=True)
        content_type = response.headers.get('Content-Type', '')
        return content_type.startswith('image')
    except:
        return False

# Apply validation (this will still take some time for large datasets)
df['valid_img'] = df['img_link'].apply(is_valid_image)

# Keep only rows with valid images
df_clean = df[df['valid_img']].reset_index(drop=True)

df_clean.to_csv("amazon_sales_clean.csv", index=False)
print(f"Cleaned dataset: {len(df_clean)} rows with valid images.")
