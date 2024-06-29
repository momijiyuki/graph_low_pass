# graphs.Graph

## class Graph (/usr/local/lib/python3.11/site-packages/pygsp/graphs/graph.py)

Graph
- __init__(self, W, gtype='unknown', lap_type='combinatorial', coords=None, plotting={}):
  1. self.logger = utils.build_logger(__name__)
      - logging.getLogger(name)が呼び出される
        - [logging docs.python](https://docs.python.org/ja/3/library/logging.html)
      - ターミナルのログ出力用みたい
  2. W = sparse.csr_matrix(W)
      - 隣接行列を疎行列に変換
  3. self.N = W.shape[0]
      - 行列の１辺の長さ
  4. self.W = sparse.lil_matrix(W)
      - 隣接行列WをLinked List形式で保存
  5. elf.Ne = self.W.nnz
      - 有向グラフであれば
        - 非零の値の数
      - 無向グラフであれば下三角行列にしてから非零の要素数を計算
        - [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.tril.html)
  6. elf.gtype = gtype
      - `__init__`の引数
  7. elf.coords = coords
      - `__init__`のオプション引数
      - 引数があるときのみ実行
  8. self.plotting = {`辞書型, 省略`}
      - プロット用のオプション
- check_weights(self):
  - `__init__`で実行
  - inf, nan, 正方行列, 対角成分の和の確認用method boolの辞書を返す
    - has_inf_val = False
    - diag_is_not_zero = False
    - is_not_square = False
    - has_nan_value = False
- set_coordinates(self, kind='spring', **kwargs):
  - ノードの座標の設定
    - グラフ描画の際に2, 3次元だったりの指定ができそう
- subgraph(self, ind):
- is_connected(self, recompute=False):
- is_directed(self, recompute=False):
  - 有向グラフか判定(Trueが有向, Falseが無向)
  - wとw.tの差があるかどうかで判定
- extract_components(self):
- compute_laplacian(self, lap_type='combinatorial'):
  - グラフラプラシアン行列の計算
  - `self.lap_type = lap_type`
    - グラフラプラシアンか対称正規化グラフラプラシアンか
  - 無向グラフのグラフラプラシアン
    - 各軸の和を対角化したもの(度数行列:D)とW(隣接行列)の差分(グラフラプラシアン)を返す
  - 無向グラフの対称正規化グラフラプラシアン
    - 重みつき度数行列の$-1/2$乗($D^{-\frac{1}{2}}$)
    - $L = I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}}$
- A(self):
  - property
  - 重みなしの隣接行列
- d(self):
  - property
  - Aの軸(axis=1で多分横向きに)に総和をとったものに要素長が1の軸を消したもの
  - 要するに重みなし隣接行列に対応する度数行列の対角成分ベクトル
- dw(self):
  - property
  - 重みつきの度数行列の対角成分
- lmax(self):
  - property
  - グラフラプラシアンの最大固有値
  - `compute_fourier_basis`で正確な値
  - `estimate_lmax`で近似値
- estimate_lmax(self, recompute=False):
  ```
  ラプラシアンの最大の固有値を推定する（キャッシュされる）。

  結果はキャッシュされ、 :attr:`lmax` プロパティでアクセスできる。

  正確な値はラプラシアンの固有値分解で得られます。
  func:`compute_fourier_basis` を参照してください。この推定は
  よりもはるかに高速です。

  パラメータ
  ----------
  recompute : boolean
      最大の固有値を強制的に再計算する。デフォルトはfalse。

  注意事項
  -----
  暗黙的に再起動されたLanczos法を大きな許容誤差で実行します、
  計算された最大固有値を1%増加させます。多くの
  PyGSPでは、Lのスペクトルを含む区間でウェーブレットカーネルを近似する必要があります。
  Lのスペクトルを含む区間でウェーブレットカーネルを近似する必要があります。
  より大きな区間を使用する唯一のコストは、より大きな区間での多項式近似が
  実際のスペクトルでは少し悪い近似になるかもしれません。
  これは非常に穏やかな効果であるため、Lのスペクトルに関する非常に厳しい境界を得る必要はない。
  のスペクトルの境界を得る必要はない。
  ```
  - `sparce.linalg.eigsh`で固有値計算
    - k=1で最大の値のみ
    - return_eigenvectors=Falseで固有値のみ
- get_edge_list(self):
  - scipyのsparseによる疎な表現化したもの
  - v_in, v_out, weights の3つのndarrayを返す
  - weights == w[v_in, v_out]が対応する
- modulate(self, f, k):
  - 信号*f*を周波数*k*に変換
  - `何に使うんだろうか`
    ```py
    nt = np.shape(f)[1]
    fm = np.kron(np.ones((1, nt)), self.U[:, k])
    fm *= np.kron(np.ones((nt, 1)), f)
    fm *= np.sqrt(self.N)
    return fm
    ```
- plot(self, **kwargs):
  - `pygsp.plotting.plot_signal`で描画
- plot_signal(self, signal, **kwargs):
  - `pygsp.plotting.plot_signal`で描画
- plot_spectrogram(self, **kwargs):
  - `pygsp.plotting.plot_spectrogram`で描画
- _fruchterman_reingold_layout(self, dim=2, k=None, pos=None, fixed=[],
                                    iterations=50, scale=1.0, center=None,
                                    seed=None):
  - ドキュメントなし
  - Fruchterman-Reingold力指向アルゴリズムを使用してノードを配置
  - 座標を固定したノードのリストを返すみたい？


## class GraphDifference (/usr/local/lib/python3.11/site-packages/pygsp/graphs/difference.py)

GraphDifference
- D(self):
  - 微分演算子
  - grad, divの計算用
- compute_differential_operator(self):
  - `combinatorial`: グラフラプラシアン
  - `normalized`: 対称正規化グラフラプラシアン
- grad(self, s):
    ```py
    return self.D.dots(s)
    ```
- div(self, s):
    ```py
    return self.D.T.dot(s)
    ```

## GraphFourier (/usr/local/lib/python3.11/site-packages/pygsp/graphs/fourier.py)

GraphFourier
- _check_fourier_properties(self, name, desc):
  - `_check_forier_properties` 内で`任意の変数`が存在しなければ `compute_fourier_basis`が呼び出されて計算される
  - その後`任意の変数`の値を返す
- U(self):
  - property
  - フーリエ基底
- e(self):
  - property
  - 固有値のベクトル
- mu(self):
  - property
  - `Coherence of the Fourier basis`
    - coherence: 類似度や一貫性といった意図？
- compute_fourier_basis(self, recompute=False):
  - `self._e, self._U = np.linalg.eigh(self.L.toarray())`
    - _e[0] = 0
    - self.Lとは
  - self._mu = np.max(np.abs(self._U))
- gft(self, s):
    ```py
    U = np.conjugate(self.U)  # True Hermitian. (Although U is often real.)
    return np.tensordot(U, s, ([0], [0]))
    ```
- igft(self, s_hat):
    ```py
    return np.tensordot(self.U, s_hat, ([1], [0]))
    ```
- translate(self, f, i):
  - 機能してない
- gft_windowed_gabor(self, s, k):
    ```py
    from pygsp import filters
    return filters.Gabor(self, k).filter(s)
    ```
- gft_windowed_(self, g, f, lowmemory=True):
  - 機能してない
- gft_windowed_normalized(self, g, f, lowmemory=True):
  - 機能してない
- _frame_matrix(self, g, normalize=False):
  - 恐らくグラフフーリエ変換の窓関数版として設計中？
    - fft方式として？
