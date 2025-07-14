# 🧠 ابزارهای مهم در پروژه‌های یادگیری ماشین

این فایل شامل دسته‌بندی کامل ابزارهایی است که در پروژه‌های واقعی یادگیری ماشین با پایتون و scikit-learn استفاده می‌شوند.

---

## 🔹 ۱. ابزارهای تبدیل داده (Transformers)

| ابزار | توضیح |
|------|-------|
| OneHotEncoder | تبدیل داده‌های متنی به بردار باینری |
| LabelEncoder | تبدیل دسته‌ها به اعداد ترتیبی |
| MinMaxScaler | مقیاس‌گذاری ویژگی‌ها بین ۰ تا ۱ |
| StandardScaler | نرمال‌سازی با میانگین صفر و واریانس یک |
| RobustScaler | مقیاس‌گذاری مقاوم در برابر داده‌های پرت |
| Normalizer | نرمال‌سازی طول بردار به ۱ |
| FunctionTransformer | اعمال توابع دلخواه مثل log یا sqrt |
| PowerTransformer | تبدیل داده‌ها به توزیع نرمال‌تر |
| ColumnTransformer | اعمال چند تبدیل روی ستون‌های مختلف |

---

## 🔹 ۲. الگوریتم‌های مدل‌سازی (Estimators)

### ✅ رگرسیون (Regression)

- LinearRegression
- Ridge, Lasso, ElasticNet
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- XGBoost, LightGBM, CatBoost
- SVR
- KNeighborsRegressor
- MLPRegressor

### ✅ طبقه‌بندی (Classification)

- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- XGBoost, CatBoost, LightGBM
- SVC
- KNeighborsClassifier
- NaiveBayes
- MLPClassifier

---

## 🔹 ۳. ابزارهای Pipeline و جریان کاری

| ابزار | کاربرد |
|------|--------|
| Pipeline | زنجیره کردن مراحل پیش‌پردازش و مدل |
| make_pipeline | ساخت ساده‌تر pipeline |
| ColumnTransformer | اجرای همزمان چند پیش‌پردازش |
| FeatureUnion | ترکیب خروجی چند تبدیل‌کننده |
| GridSearchCV | جستجوی پارامترهای بهینه مدل |
| RandomizedSearchCV | جستجوی سریع‌تر پارامترها |
| cross_val_score | اعتبارسنجی متقاطع |
| train_test_split | تقسیم داده به آموزش و تست |
| KFold, StratifiedKFold | تقسیم‌های پیشرفته‌تر برای اعتبارسنجی |

---

## 🔹 ۴. ذخیره و بارگذاری مدل

| ابزار | توضیح |
|------|-------|
| joblib.dump() / joblib.load() | ذخیره و بازیابی مدل یا pipeline بهینه |
| pickle | ذخیره مدل به صورت باینری (برای پروژه‌های سبک) |

---

## 🔹 ۵. ارزیابی مدل

### 📏 برای رگرسیون:
- mean_squared_error — میانگین مربعات خطا
- mean_absolute_error — میانگین خطای مطلق
- r2_score — ضریب تعیین

### 🧮 برای طبقه‌بندی:
- accuracy_score — دقت کلی
- precision_score, recall_score, f1_score — معیارهای عملکرد
- confusion_matrix — ماتریس خطا
- classification_report — گزارش کامل عملکرد

---

## 🔹 ۶. ابزارهای تصویری و تحلیل داده

| ابزار | کاربرد |
|------|--------|
| matplotlib | ترسیم نمودارهای پایه |
| seaborn | نمودارهای آماری پیشرفته |
| plotly | نمودارهای تعاملی و سه‌بعدی |
| pandas_profiling یا ydata-profiling | ساخت گزارش کامل داده‌ها |
| missingno | نمایش بصری مقادیر گمشده |
| sweetviz | آنالیز خودکار مقایسه‌ای داده‌ها |
| plot_partial_dependence | بررسی تأثیر ویژگی‌ها بر پیش‌بینی مدل |

---

## 🔹 ۷. ابزارهای پیشرفته

| ابزار | کاربرد |
|------|--------|
| SHAP | تفسیر خروجی مدل‌های پیچیده مثل XGBoost |
| LIME | تفسیر مدل به صورت محلی |
| Optuna, Hyperopt | بهینه‌سازی پارامترها |
| MLflow, Weights & Biases | ثبت و بررسی آزمایش‌ها |
| scikit-optimize | جستجوی پارامتر با روش بیزی (Bayesian) |

---

## 🔹 ۸. مهندسی ویژگی (Feature Engineering)

| ابزار | کاربرد |
|------|--------|
| PolynomialFeatures | تولید ویژگی‌های چندجمله‌ای |
| Binarizer | تبدیل اعداد به صفر و یک |
| KBinsDiscretizer | دسته‌بندی مقادیر عددی به بازه‌ها |
| SelectKBest, RFE | انتخاب ویژگی‌های مهم |
| VarianceThreshold | حذف ویژگی‌های با واریانس پایین |

---

## ✅ نکته آخر

> ❗️ لازم نیست همه این ابزارها رو حفظ باشید، فقط بدونید کی و چرا ازشون استفاده می‌شه.
