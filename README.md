# Interface gráfica de Visão Computacional

Aplicação em CustomTkinter sobre a biblioteca `vc` (filtros, geometria, câmera, etc.).

## Executar

```
pip install -r requirements.txt
python app.py
```

Testes: `python -m unittest discover -s tests`

## Aba "Formação de Imagem" (Atividade 01)

Projeção de um cubo (vértices P1–P8 no mundo) com modelo **perspectiva** ou **ortográfica escalada**,
sensor 640×480, 800×600 ou 1280×720, parâmetros intrínsecos (fx, fy, cx, cy, skew, matriz K),
escala (ortográfica) e parâmetros extrínsecos (translação e rotação em X, Y, Z). A imagem é atualizada a
cada alteração; use os campos, os sliders ou o mouse sobre a imagem:

- botão esquerdo + arrastar: gira em X/Y (Shift: gira em Z)
- botão direito + arrastar: transla em X/Y
- roda do mouse: aproxima/afasta (tz)

Equações (código em `vc/formacao_imagem.py`): `Xc = R·Xw + t`, `R = Rz·Ry·Rx`;
perspectiva `x̃ = K·Xc`, `u = x̃₁/x̃₃`, `v = x̃₂/x̃₃`; ortográfica `u = s·Xc + cx`, `v = s·Yc + cy`
com (cx, cy) no centro do sensor. Na perspectiva, cx e cy não mudam ao trocar a resolução.
