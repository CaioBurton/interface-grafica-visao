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

## Atividade 02 - Blending com pirâmides

`vc/piramides.py`: `reduzir`/`expandir` (núcleo binomial 5x5), `piramide_gaussiana`, `piramide_laplaciana`,
`colapsar`, `blending` (Burt e Adelson, 1983) e `justaposicao`. Para cada nível,
`L_blend = Gm·La + (1−Gm)·Lb`, com `Gm` a pirâmide Gaussiana da máscara; a pirâmide é então colapsada.

`python blending.py [pasta_imagens] [pasta_saida] [niveis]` gera em `resultados_blending/` o blending,
a justaposição direta, a comparação, o zoom da emenda e as pirâmides.

Na interface, a aba **Blending (Pirâmides)** carrega por padrão a maçã, a laranja e a máscara sugeridas;
é possível escolher outras imagens A/B (a B é redimensionada para o tamanho da A), uma máscara em arquivo
ou uma máscara automática (metade esquerda/direita), e o número de níveis. As visões são: comparação,
blending, justaposição, zoom da emenda e pirâmides (salvar com "Salvar visualização").
