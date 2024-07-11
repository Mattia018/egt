import torch

def check_gpu():
    if torch.cuda.is_available():
        print("La GPU è disponibile per PyTorch!")
        print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")
        print(f"Nome della GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("La GPU non è disponibile per PyTorch. Verrà utilizzata la CPU.")

if __name__ == "__main__":
    check_gpu()
