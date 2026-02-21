import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def parse_multi_aspect_xml(file_path):
    """
    این تابع فایل XML را می‌گیرد و به ازای هر Aspect Term یک سطر می‌سازد
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []

    for sentence in root.findall('sentence'):
        sentence_id = sentence.get('id')
        text_node = sentence.find('text')
        if text_node is None: continue
        text = text_node.text
        
        aspect_terms = sentence.find('aspectTerms')
        
        # اگر جمله جنبه‌ای داشت، استخراجش می‌کنیم
        if aspect_terms is not None:
            for aspect in aspect_terms.findall('aspectTerm'):
                term = aspect.get('term')
                polarity = aspect.get('polarity')
                
                # نادیده گرفتن کدهای conflict برای تمرکز بر مثبت/منفی/خنثی
                if polarity in ['positive', 'negative', 'neutral']:
                    data.append({
                        'id': sentence_id,
                        'text': text,
                        'aspect': term,
                        'sentiment': polarity
                    })
    
    return pd.DataFrame(data)

# ۱. استخراج داده‌ها از فایل اصلی که فرستادی
file_name = 'Restaurants_Train_v2.xml' # یا هر فایل دیگری که مد نظر است
df = parse_multi_aspect_xml(file_name)

# ذخیره خروجی تمیز شده در یک فایل CSV برای مراحل بعد
df.to_csv('restaurant_cleaned_data.csv', index=False)

print(f"✅ استخراج با موفقیت انجام شد. تعداد کل رکوردها: {len(df)}")
print(df.head(10))

# --- بخش تحلیل (EDA) برای گرفتن نمره کامل ---

# الف) نمودار توزیع احساسات
plt.figure(figsize=(7, 5))
sns.countplot(x='sentiment', data=df, palette='magma')
plt.title('Distribution of Sentiments (Restaurant)')
plt.show()

# ب) هیستوگرام طول جملات (تعداد کلمات)
df['sentence_len'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(7, 5))
sns.histplot(df['sentence_len'], bins=20, kde=True, color='blue')
plt.title('Histogram of Sentence Lengths')
plt.show()

# ج) ابر کلمات برای جنبه‌ها (Aspects)
aspect_text = " ".join(df['aspect'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(aspect_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Aspect Terms')
plt.show()