(ns mlalgorithms.protocols)

(defprotocol PModel
  (fit [this X y])
  (predict [this X]))

(defprotocol PGAN
  (init [this])
  (build-generator [this optimizer loss-function])
  (build-discriminator [this optimizer loss-function])
  (train [this n-epochs batch-size save-interval])
  (save-imgs [this epoch]))
