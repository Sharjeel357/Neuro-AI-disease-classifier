import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import google.generativeai as genai

# Inject API key
genai.configure(api_key="AIzaSyDn5yK2_2pYMId3bpFlAf0LkWoJ7dvEcqM")

# ------------------ Model Definitions ------------------

class AlzheimerCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class BrainHemorrhageCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StrokeCTCNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class BrainTumorCNN(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ Load Models ------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alz_model = AlzheimerCNN()
alz_model.load_state_dict(torch.load(r"C:\Users\HP\OneDrive\Desktop\trained model\alzmodel.pth", map_location=device))
alz_model.eval()

brainh_model = BrainHemorrhageCNN()
brainh_model.load_state_dict(torch.load(r"C:\Users\HP\OneDrive\Desktop\trained model\brainham.pth", map_location=device))
brainh_model.eval()

brainst_model = StrokeCTCNN()
brainst_model.load_state_dict(torch.load(r"C:\Users\HP\OneDrive\Desktop\trained model\brainst_model.pth", map_location=device))
brainst_model.eval()

braint_model = BrainTumorCNN()
braint_model.load_state_dict(torch.load(r"C:\Users\HP\OneDrive\Desktop\trained model\braintmodel.pth", map_location=device))
braint_model.eval()

# ------------------ Recommendation Mapping ------------------

recommendation_dict = {
    "Non Demented": "No signs of Alzheimer's detected. Maintain a healthy brain with mental activities and regular checkups(recomend you to use the gemini chatbot).",
    "Demented": "Signs of dementia detected. Consult a neurologist for proper diagnosis and treatment options.(recomend you to use the gemini chatbot)",
    "Normal": "Brain scan appears normal. Stay consistent with health checks.(recomend you to use the gemini chatbot)",
    "Hemorrhagic": "Hemorrhage detected. Seek immediate medical attention — may require surgery or ICU.(recomend you to use the gemini chatbot)",
    "Bleeding": "Bleeding stroke identified. Emergency treatment may be necessary.(recomend you to use the gemini chatbot)",
    "Ischemia": "Ischemic stroke detected. Treatment may include clot-busting medication.(recomend you to use the gemini chatbot)",
    "Glioma": "Glioma tumor found. Requires MRI follow-up and oncology consultation.(recomend you to use the gemini chatbot)",
    "Meningioma": "Meningioma detected. Often benign but may need surgical evaluation.(recomend you to use the gemini chatbot)",
    "No Tumor": "No brain tumor detected. Continue routine monitoring.(recomend you to use the gemini chatbot)",
    "Pituitary": "Pituitary tumor detected. Hormonal and visual exams recommended.(recomend you to use the gemini chatbot)"
}

# ------------------ Prediction Function ------------------

def predict(disorder, image):
    img = Image.open(image).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    if disorder == "Alzheimer":
        outputs = alz_model(img_tensor)
        class_names = ["Non Demented", "Demented"]

    elif disorder == "Brain Hemorrhage":
        outputs = brainh_model(img_tensor)
        class_names = ["Normal", "Hemorrhagic"]

    elif disorder == "Brain Stroke":
        outputs = brainst_model(img_tensor)
        class_names = ["Bleeding", "Ischemia", "Normal"]

    elif disorder == "Brain Tumor":
        outputs = braint_model(img_tensor)
        class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_label = class_names[pred_idx]
    recommendation = recommendation_dict.get(pred_label, "No recommendation available.")

    return f"Prediction: {pred_label}\n\nRecommendation: {recommendation}"

# ------------------ Gemini Chatbot Function ------------------

model = genai.GenerativeModel("models/gemini-1.5-flash")

def chat_with_gemini(user_input, history=[]):
    prompt = "You are a highly experienced neurologist. Help answer questions related to brain diseases like stroke, dementia, tumors, Alzheimer’s, etc. and you answer questions short and simple to make a non medical field based person understand it better and easily\n"
    convo = model.start_chat(history=[])
    convo.send_message(prompt + user_input)
    return convo.last.text

# ------------------ Gradio Interface ------------------

with gr.Blocks(theme=gr.themes.Base(), title="Brain Neurologist App") as demo:
    with gr.Tab("Brain Scan Predictor"):
        gr.Markdown("## 🧠 MRI/CT Brain Disorder Detection")
        disorder_input = gr.Dropdown(["Alzheimer", "Brain Hemorrhage", "Brain Stroke", "Brain Tumor"], label="Select Disorder")
        image_input = gr.Image(type="filepath", label="Upload Brain Scan")
        output_text = gr.Textbox(label="Prediction and Recommendation")
        submit_btn = gr.Button("Predict")
        submit_btn.click(fn=predict, inputs=[disorder_input, image_input], outputs=output_text)

    with gr.Tab("🧑‍⚕️ Dr.Neuro"):
        gr.Markdown("## Gemini powered Neurologist Chatbot")
        chatbot = gr.ChatInterface(fn=chat_with_gemini)

demo.launch(share=True)
