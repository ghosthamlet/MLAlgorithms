(ns mlalgorithms.protocols)

(defprotocol PModel
  (fit [this X y] [this X y n-epochs batch-size])
  (predict [this X]))

(defprotocol POptimizer
  (opt-grad [this w grad-wrt-w]))

(defprotocol PLoss
  (loss [this y y-pred])
  (loss-grad [this y y-pred])
  (acc [this y y-pred]))

(defprotocol PGAN
  (init-gan-vars [this])
  (init-gan [this])
  (build-generator [this optimizer loss-function])
  (build-discriminator [this optimizer loss-function])
  (train [this n-epochs batch-size save-interval samples])
  (save-imgs [this epoch]))
