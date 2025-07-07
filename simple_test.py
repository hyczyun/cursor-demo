import streamlit as st

st.title("简单测试")
st.write("如果你看到这个页面，说明Streamlit工作正常！")

try:
    from algorithms.fault_detection import FaultDetector
    st.success("FaultDetector 导入成功！")
except Exception as e:
    st.error(f"导入错误: {e}")

st.write("测试完成！") 