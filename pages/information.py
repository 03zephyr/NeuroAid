import streamlit as st

st.title("NeuroAid 🧠")
st.text("")
st.header("Input Features Information")
st.text("")
st.write("### Mini-Mental State Examination (MMSE)")
st.write("""
The Mini-Mental State Examination (MMSE) is a widely used screening tool for assessing cognitive impairment. It evaluates five key areas of cognitive function: orientation, registration (immediate memory), attention and calculation, recall, and language. The test is quick to administer, taking only 5-10 minutes, and provides a score out of 30.""")
st.write("##### Scoring and Interpretation:")
st.write("""
- **25-30**: Normal cognitive function
- **21-24**: Mild cognitive impairment
- **10-20**: Moderate cognitive impairment
- **<10**: Severe cognitive impairment
""")
st.write("While the MMSE is not a diagnostic tool, it helps identify individuals who may need further evaluation for conditions like Alzheimer's disease or dementia.")

st.text("")
st.write("### Memory Complaints")
st.write("""
Memory complaints refer to a subjective awareness of difficulties in remembering information, which may be an early indicator of cognitive decline or Alzheimer's disease. This feature is recorded as a 'Yes' or 'No' response, reflecting whether the individual has noticed issues with their memory.
""")
st.write("##### Key Points:")
st.write("""
- Memory complaints are common in older adults and may signal self-awareness of degenerative processes.
- They can occur in individuals with or without diagnosed dementia, often preceding objective signs of cognitive impairment.
- While not definitive on their own, memory complaints should be taken seriously and may warrant further evaluation.
""")

st.text("")
st.write("### Behavioral Problems")
st.write("""
Behavioral problems refer to changes in a person's actions, emotions, or reactions that can occur due to Alzheimer's disease or other dementias. These changes are often challenging for caregivers and may include agitation, aggression, anxiety, restlessness, or social withdrawal. This feature is recorded as a 'Yes' or 'No' response, indicating whether such issues are present.
""")
st.write("##### Key Points:")
st.write("""
- Behavioral problems can emerge early or progress with the disease.
- Common examples include irritability, wandering, repetitive actions, or resistance to care.
- These behaviors often result from confusion, frustration, or unmet needs and may vary in severity.
- Identifying triggers and providing appropriate support can help manage these challenges effectively.
""")

st.text("")
st.write("### Activities of Daily Living (ADL)")
st.write("""
Activities of Daily Living (ADL) refer to essential tasks required for independent living, such as eating, bathing, dressing, toileting, transferring, and maintaining continence. The ADL score ranges from 0 to 10, where higher scores indicate greater independence.
""")
st.write("##### Scoring and Interpretation:")
st.write("""
- **8-10**: Independent in most or all activities.
- **4-7**: Moderate impairment, requiring assistance with some activities.
- **0-3**: Severe impairment, dependent on others for most or all activities.
""")
st.write("This score helps assess functional ability and determine the level of support an individual may need.")


st.text("")
st.write("### Functional Assessment")
st.write("""
Functional assessment evaluates an individual's ability to perform daily activities and maintain independence. It is commonly used in dementia care to assess the level of functional impairment. The score ranges from 0 to 10, where higher scores indicate better functional ability.
""")
st.write("##### Scoring and Interpretation:")
st.write("""
- **8-10**: Minimal or no functional impairment, able to perform most tasks independently.
- **4-7**: Moderate functional impairment, requiring assistance with some activities.
- **0-3**: Severe functional impairment, dependent on others for most or all activities.
""")
st.write("This score helps caregivers and healthcare professionals monitor changes over time and develop appropriate care plans.")
