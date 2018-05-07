(ns mlalgorithms.protocols)

(defprotocol PModel
  (fit [this X y])
  (predict [this X]))
