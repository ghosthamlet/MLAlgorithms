(ns mlalgorithms.deep-learning.neural-network
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.stats :as ms]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.selection :as sel]
            [clojure.pprint :as pp]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.deep-learning.layers :as layer]
            [mlalgorithms.deep-learning.loss-functions :as loss]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defprotocol PNeuralNetwork
  (init-nn [this])
  (set-trainable [this trainable])
  (add-layer [this layer])
  (test-on-batch [this X y])
  (train-on-batch [this X y])
  (forward-pass* [this X training])
  (backward-pass* [this loss-grad])
  (summary [this name]))

(defpyrecord NeuralNetwork
  [optimizer (loss-function)
   (validation-data) (layers [])
   (n-epochs) (batch-size)
   (errors {:training [] :validation []}) (val-set)]
  PNeuralNetwork
  (init-nn [this]
           (assoc this
                  :val-set (when validation-data
                             {:X (validation-data 0)
                              :y (validation-data 1)})))

  (set-trainable [this trainable]
                 (map #(assoc %
                              :trainable trainable)
                      layers))

  (add-layer [this layer]
             (update-in this
                        [:layers]
                        conj (layer/initialize (if (seq layers)
                                                      ;; If this is not the first layer added then set the input shape
                                                      ;; to the output shape of the last added layer
                                                      (layer/set-input-shape layer
                                                                             (layer/output-shape (last layers)))
                                                      layer)
                                                    optimizer)))

  (test-on-batch [this X y]
                 (let [y-pred (:output (forward-pass* this X false))
                       loss (ms/mean (loss/loss loss-function y y-pred))
                       acc (loss/acc loss-function y y-pred)]
                   [loss acc]))

  (train-on-batch [this X y]
                  (let [y-pred (forward-pass* this X true)
                        loss (ms/mean (loss/loss loss-function y y-pred))
                        acc (loss/acc loss-function y y-pred)
                        loss-grad (loss/grad loss-function y y-pred)]
                    [(backward-pass* this loss-grad) loss acc]))

  (forward-pass* [this X training]
                 (reduce #(do (prn (type %2)) (layer/forward-pass %2 (:output %1) training))
                         {:output X}
                         layers))

  (backward-pass* [this loss-grad]
                  (reduce #(layer/backward-pass %2 %1)
                          loss-grad
                          (reverse layers)))

  (summary [this name]
           (prn (str "Model: " name))
           (prn (str "Input Shape: " (:input-shape (layers 0))))
           (loop [[layer & n] layers
                  table-data [["Layer Type" "Parameters" "Output Shape"]]
                  tot-params 0]
             (if (nil? layer)
               (do
                 (pp/pprint table-data)
                 (prn (str "Total Parameters: \n" tot-params)))
               (let [layer-name (str (type layer))
                     params (layer/parameters layer)
                     out-shape (layer/output-shape layer)]
                 (recur n
                        (conj table-data
                              [layer-name (str params) (str out-shape)])
                        (+ tot-params params))))))

  p/PModel
  (fit [this X y]
       (let [n-samples ((shape X) 0)
             batch-error-fn #(loop [[i & is] (range 0 n-samples batch-size)
                                    batch-error []]
                               (if (nil? i)
                                 batch-error
                                 (let [[begin end] [i (min (+ i batch-size)
                                                           n-samples)]
                                       X-batch (sel/sel X (sel/irange begin end))
                                       y-batch (sel/sel y (sel/irange begin end))]
                                   (recur is
                                          (conj batch-error
                                                ((train-on-batch this
                                                                 X-batch
                                                                 y-batch) 0))))))]
         (loop [[i & is] (range n-epochs)
               errors errors]
          (if (nil? i)
            (assoc this
                   :errors errors)
            (recur is
                   (assoc errors
                          :training (conj (:training errors)
                                          (ms/mean (batch-error-fn)))
                          :validation (if val-set
                                        (conj (:validation errors)
                                              ((test-on-batch this
                                                              (:X val-set)
                                                              (:y val-set)) 0))
                                        (:validation errors))))))))

  (predict [this X]
           (forward-pass* this X false)))
