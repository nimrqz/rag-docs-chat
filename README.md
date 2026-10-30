# Chat com documentos (RAG)

Você sobe um PDF, Markdown ou TXT e faz perguntas. A resposta vem com o número do trecho usado (`[1]`, `[2]`), e se a informação não está no documento ele responde que não encontrou.

Como funciona: o texto é dividido em pedaços, os pedaços mais parecidos com a pergunta são buscados (TF-IDF) e enviados para o Claude responder só com base neles.

## Como rodar

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
streamlit run app.py
python -m eval.run_eval      # testa só a busca, sem usar a API
python -m pytest tests
```

Sem subir nada, o app usa um documento de exemplo (política de home office de uma empresa fictícia).

## Resultado

Com 12 perguntas que eu escrevi sobre o documento de exemplo, o trecho certo apareceu entre os 3 primeiros em 100% dos casos (MRR 0,96). É um teste pequeno e as perguntas são parecidas com o texto, então não dá para generalizar.

## Limitações

- Usei TF-IDF em vez de embeddings porque a API da Claude não tem embeddings. A busca erra mais quando a pergunta usa palavras diferentes das do documento.
- PDFs escaneados (só imagem) não funcionam, porque o texto é extraído com o `pypdf`.

## Próximos passos

- Rodar `python -m eval.run_eval --answers` para medir também a qualidade das respostas
- Trocar o TF-IDF por embeddings
