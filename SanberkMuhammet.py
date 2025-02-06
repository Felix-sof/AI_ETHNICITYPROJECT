import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from tqdm import tqdm
import cv2
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision için gerekli modüller
from multiprocessing import freeze_support  # Windows'ta multiprocessing için gerekli
import torchvision  # Torchvision beta uyarılarını kapatmak için
import threading  # Threading modülü eklendi
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow uyarılarını kapatma
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow uyarılarını kapatır
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN uyarılarını kapatır

# Torchvision beta uyarılarını kapatma
torchvision.disable_beta_transforms_warning()

# Veri dengesizliğini gidermek için her ırktan eşit sayıda örnek seçen fonksiyon
def balance_dataset(dataframe, label_column='race'):
    min_count = dataframe[label_column].value_counts().min()
    balanced_data = dataframe.groupby(label_column, group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    return balanced_data

# Morph veri seti yolu
morph_path = r'C:\Users\sanbe\OneDrive\Masaüstü\data\Morph\CD1\Album1'
morph_file = r'C:\Users\sanbe\OneDrive\Masaüstü\data\Morph\CD1\MORPH_Album1_PGMDATA.xls'

# Morph veri setini yükle
morph_data = pd.read_excel(morph_file)

# Irk etiketlerini düzeltme
race_corrections = {
    'African-American.Black': 'Black',
    'Caucasian.White': 'White',
    'Latino.Hispanic': 'Latino',
    'Asian': 'Asian',
    'Middle Eastern': 'Middle Eastern',
    'Southeast Asian': 'Asian',
    'Other': 'Other',
}
morph_data['race'] = morph_data['race'].map(race_corrections).fillna(morph_data['race'])

# Fotoğraf yollarını düzeltme
morph_data['photo'] = morph_data['photo'].apply(
    lambda x: os.path.abspath(os.path.join(morph_path, os.path.basename(x.replace('.PGM', '.jpg'))))
)
morph_data = morph_data[morph_data['photo'].apply(os.path.exists)]

# UTKFace veri seti yolları
utkface_paths = [
    r'C:\Users\sanbe\OneDrive\Masaüstü\data\UTKFace\part1',
    r'C:\Users\sanbe\OneDrive\Masaüstü\data\UTKFace\part2',
    r'C:\Users\sanbe\OneDrive\Masaüstü\data\UTKFace\part3'
]

# UTKFace veri setini yükle
utk_data = []
for path in utkface_paths:
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            try:
                parts = file.split('_')
                if len(parts) >= 3:
                    race = int(parts[2])
                    utk_data.append({'photo': os.path.join(path, file), 'race': race})
            except ValueError:
                continue

# UTKFace veri setini DataFrame'e dönüştür
utk_df = pd.DataFrame(utk_data)

# Sınıf etiketlerini eşle
race_mapping = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian'}
utk_df['race'] = utk_df['race'].map(race_mapping)

# UTKFace veri setini dengele
utk_df = balance_dataset(utk_df)

# FairFace veri seti yolları
fairface_train_csv = r'C:\Users\sanbe\OneDrive\Masaüstü\data\FairFace\fairface_label_train.csv'
fairface_val_csv = r'C:\Users\sanbe\OneDrive\Masaüstü\data\FairFace\fairface_label_val.csv'

# FairFace eğitim ve doğrulama verisini yükle
fairface_train_df = pd.read_csv(fairface_train_csv)
fairface_val_df = pd.read_csv(fairface_val_csv)

# FairFace etiketlerini ve fotoğraf yollarını düzeltme
fairface_train_df['photo'] = fairface_train_df['file'].apply(
    lambda x: os.path.abspath(os.path.join(r'C:\Users\sanbe\OneDrive\Masaüstü\data\FairFace\train', os.path.basename(x)))
)
fairface_val_df['photo'] = fairface_val_df['file'].apply(
    lambda x: os.path.abspath(os.path.join(r'C:\Users\sanbe\OneDrive\Masaüstü\data\FairFace\val', os.path.basename(x)))
)

# FairFace etiketlerini düzeltme
fairface_train_df['race'] = fairface_train_df['race'].map(race_corrections)
fairface_val_df['race'] = fairface_val_df['race'].map(race_corrections)

# Eksik veya hatalı verileri temizleme
fairface_train_df = balance_dataset(fairface_train_df)
fairface_val_df = balance_dataset(fairface_val_df)

# CNSIFD veri seti yolu
cnsifd_path = r'C:\Users\sanbe\OneDrive\Masaüstü\data\CNSIFD Indian\cnsifd_faces'

# CNSIFD veri setini yükle
cnsifd_data = []
for file in os.listdir(cnsifd_path):
    if file.endswith('.bmp'):
        cnsifd_data.append({'photo': os.path.join(cnsifd_path, file), 'race': 'Indian'})

# CNSIFD'yi DataFrame'e dönüştür
cnsifd_df = pd.DataFrame(cnsifd_data)

# Veri setlerini birleştir
combined_data = pd.concat([morph_data, utk_df, fairface_train_df, fairface_val_df, cnsifd_df], ignore_index=True)

# Her ırktan kaç veri kullanıldığını hesapla ve yazdır
race_counts = combined_data['race'].value_counts()
print("Her ırktan kullanılan veri sayısı:")
print(race_counts)

# Sınıf etiketlerini numaralandır
race_to_label = {race: idx for idx, race in enumerate(combined_data['race'].unique())}
combined_data['label'] = combined_data['race'].map(race_to_label)

# Eğitim ve test veri setlerini ayır
train_data, test_data = train_test_split(
    combined_data, test_size=0.2, stratify=combined_data['label'], random_state=42
)

# Veri seti sınıfı
class EthnicityDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['photo']
        label = self.dataframe.iloc[idx]['label']
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            warnings.warn(f"Resim yüklenemedi: {img_path}, Hata: {e}")
            return None  # Hatalı örnekleri atla

# Görüntü dönüşümleri
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Hatalı örnekleri filtrelemek için collate_fn
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # None değerlerini filtrele
    return torch.utils.data.dataloader.default_collate(batch)

# Veri yükleyiciler
train_loader = DataLoader(EthnicityDataset(train_data, transform=data_transforms), batch_size=16, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(EthnicityDataset(test_data, transform=data_transforms), batch_size=16, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

# Vision Transformer modelini yükle
model_name = 'google/vit-base-patch16-224-in21k'
id2label = {idx: label for label, idx in race_to_label.items()}
label2id = {label: idx for idx, label in race_to_label.items()}

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(race_to_label),
    id2label=id2label,
    label2id=label2id
)

# Cihaz kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Eğitim parametreleri
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Mixed Precision için GradScaler
scaler = GradScaler()  # device='cuda' parametresi gerekli değil

# Modeli kaydetme fonksiyonu
def save_model(model, save_path="model.pth"):
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

# Modeli yükleme fonksiyonu
def load_model(model, load_path="model.pth"):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    print(f"Model yüklendi: {load_path}")

# Eğitim fonksiyonu
def train_model(model, dataloader, criterion, optimizer, device, epochs=3, save_dir="training_results"):
    train_losses = []
    train_accuracies = []

    # Sonuçların kaydedileceği klasör
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            valid_mask = labels != -1
            images, labels = images[valid_mask], labels[valid_mask]
            if len(labels) == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed Precision Training
            with autocast():  # autocast kullanımı
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Her epoch sonunda ara sonuçları kaydet
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Loss', marker='o')
        plt.plot(train_accuracies, label='Accuracy', marker='o')
        plt.title(f'Training Loss and Accuracy (Epoch {epoch+1})')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()

        # Grafik kaydetme
        epoch_plot_path = os.path.join(save_dir, f'epoch_{epoch+1}_results.png')
        plt.savefig(epoch_plot_path)
        plt.close()  # Grafik bellekten silinir, işlem devam eder

        print(f"Epoch {epoch+1} sonuç grafiği kaydedildi: {epoch_plot_path}")

    # Sonuçların genel grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Loss', marker='o')
    plt.plot(train_accuracies, label='Accuracy', marker='o')
    plt.title('Training Loss and Accuracy Over All Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    # Genel grafik kaydetme
    overall_plot_path = os.path.join(save_dir, 'overall_results.png')
    plt.savefig(overall_plot_path)
    plt.close()

    print(f"Genel sonuç grafiği kaydedildi: {overall_plot_path}")

    # Modeli kaydet
    save_model(model, os.path.join(save_dir, "model.pth"))

# Hata analizi ve karmaşıklık matrisi ROC eklenmiş fonksiyon
def evaluate_model_with_roc(model, dataloader, device, id2label):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(id2label.values())))

    # Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(id2label))
    plt.xticks(tick_marks, id2label.values(), rotation=45)
    plt.yticks(tick_marks, id2label.values())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join("training_results", "confusion_matrix.png"))
    plt.show(block=False)  # Grafiği göster, ancak programı durdurma
    plt.pause(5)  # Grafiği 5 saniye göster
    plt.close()  # Grafiği kapat

    # ROC ve AUC Hesaplama
    all_labels_one_hot = np.eye(len(id2label))[all_labels]
    auc_scores = {}
    plt.figure(figsize=(10, 8))
    for i, label in id2label.items():
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], np.array(all_probs)[:, i])
        auc = roc_auc_score(all_labels_one_hot[:, i], np.array(all_probs)[:, i])
        auc_scores[label] = auc
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

    # AUC Scores
    print("AUC Scores:")
    for label, auc in auc_scores.items():
        print(f"{label}: {auc:.4f}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join("training_results", "roc_curves.png"))
    plt.show(block=False)  # Grafiği göster, ancak programı durdurma
    plt.pause(5)  # Grafiği 5 saniye göster
    plt.close()  # Grafiği kapat

# Kamera işlemlerini ayrı bir thread'de çalıştıran sınıf
class CameraThread(threading.Thread):
    def __init__(self, model, device, data_transforms, id2label):
        threading.Thread.__init__(self)
        self.model = model
        self.device = device
        self.data_transforms = data_transforms
        self.id2label = id2label
        self.running = False

    def run(self):
        # Kamera başlat
        self.cap = cv2.VideoCapture(0)  # 0, varsayılan kamera için
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            return

        print("Kamera açıldı. Çıkmak için 'q' tuşuna basın. Tahmin almak için 'c' tuşuna basın.")
        self.running = True

        while self.running:
            # Kameradan bir kare al
            ret, frame = self.cap.read()
            if not ret:
                print("Kameradan görüntü alınamadı!")
                break

            # Görüntüyü ekranda göster
            cv2.imshow("Kamera Tahmini", frame)

            # Klavye girdisini kontrol et
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # 'c' tuşuna basıldığında tahmin yap
                self.predict_race(frame)
            elif key == ord('q'):  # 'q' tuşuna basıldığında çık
                print("Kamera kapatılıyor...")
                self.running = False

        # Kamera kaynaklarını serbest bırak
        self.cap.release()
        cv2.destroyAllWindows()

    def predict_race(self, frame):
        try:
            # Görüntüyü PIL Image'a çevir
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Görüntüyü işle ve modele gönder
            image = self.data_transforms(pil_img).unsqueeze(0).to(self.device)

            # Model tahminini al
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image).logits
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # Tahmin sonuçlarını sırala
            predictions = {self.id2label[idx]: prob for idx, prob in enumerate(probabilities)}
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            top_prediction = sorted_predictions[0]

            # Tahmin sonuçlarını yazdır
            print("\nTahmin edilen ırklar ve yüzdeler:")
            for race, prob in sorted_predictions:
                print(f"{race}: {prob * 100:.2f}%")

            # Görüntüye tahmin sonucunu ekle
            overlay_text = f"{top_prediction[0]}: {top_prediction[1] * 100:.2f}%"
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Kamera Tahmini", frame)

        except Exception as e:
            print(f"Görüntü işlenirken bir hata oluştu: {e}")

# Ana kod bloğu
if __name__ == '__main__':
    freeze_support()  # Windows'ta multiprocessing için gerekli

    # Modeli yükle (eğitim adımını atla)
    load_model(model, os.path.join("training_results", "model.pth"))

    # Kamera thread'ini başlat
    camera_thread = CameraThread(model, device, data_transforms, id2label)
    camera_thread.start()

    # Ana programın devam etmesini sağla
    try:
        while camera_thread.is_alive():
            time.sleep(1)  # Ana programın sürekli çalışmasını sağla
    except KeyboardInterrupt:
        print("Program sonlandırılıyor...")

    # Kamera thread'ini durdur
    camera_thread.running = False
    camera_thread.join()

    # Test veri setini değerlendir
    evaluate_model_with_roc(model, test_loader, device, id2label)

    # Kameradan çıkıldıktan sonra tabloları göster
    print("Eğitim ve değerlendirme sonuçları 'training_results' klasöründe kaydedildi.")
    print("Aşağıdaki grafikler gösterilecek:")

    # Kaydedilen grafikleri aç
    training_results_dir = "training_results"
    if os.path.exists(training_results_dir):
        for filename in os.listdir(training_results_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(training_results_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.imshow(filename, img)
                    cv2.waitKey(5000)  # Her bir grafik için 5 saniye bekler
                    cv2.destroyAllWindows()
                else:
                    print(f"{filename} dosyası okunamadı.")
    else:
        print(f"{training_results_dir} klasörü bulunamadı.")  