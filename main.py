import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time

st.title("Welcome Zhang huangui")

df=pd.DataFrame(
    {
        '1列目': [1,2, 3, 4],
        '2列目': [10,20, 30, 40]
    }
)

#st.write(df)

st.dataframe(df.style.highlight_max(axis=0))

"""
# 章
## 節
### 項

```python
import streamlit as st
import numpy as np
import pandas asa pd
```

"""

df = pd.DataFrame(
  np.random.rand(20, 3),
  columns=['a', 'b', 'c']
)

st.write(df)

st.line_chart(df)

st.checkbox("show Image")

if st.checkbox("show Image", key ='chk1'):
  img = Image.open("idphoto.jpg")
  st.image(img, caption='choukanki', use_column_width=True)

option = st.selectbox(
  "あなたが好きな数字をおしえてください",
  list(range(1, 10))
)

"あなたの好きな数字は", option, "です。"

st.write("Interactive Widgets")

text = st.sidebar.text_input("あなたの興味を教えてください")
st.write("""あなたの興味は""", text)

condition = st.sidebar.slider("あなたの調子をおしえてください", 0, 100, 50)
"コンディション", condition

"プログレスバー表示"

interation = st.empty()
bar = st.progress(0)

for i in range(100):
  interation.text(f'Iteration {i + 1}')
  bar.progress(i + 1)
  time.sleep(0.1)

"Done!!!!!!!!"