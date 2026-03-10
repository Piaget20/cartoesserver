import requests
import numpy as np

URL = "http://localhost:5000"

# 1. Testar Extração
print("--- Testando Extração ---")
with open("IMG_3011.JPG", "rb") as f:
    res = requests.post(f"{URL}/extrair", files={"foto": f})

if res.status_code == 200:
    dados = res.json()
    embedding_1 = dados['embedding']
    print("Sucesso! Embedding extraído.")
    
    # 2. Testar Comparação
    print("\n--- Testando Comparação ---")
    payload = {
        "novo": embedding_1,
        "conhecidos": [embedding_1] # Comparando com ele mesmo para testar 100%
    }
    res_comp = requests.post(f"{URL}/comparar", json=payload)
    print("Resultado da Similitude:", res_comp.json())
else:
    print("Erro na extração:", res.json())