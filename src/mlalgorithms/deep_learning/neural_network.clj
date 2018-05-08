(ns mlalgorithms.deep-learning.neural-network
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.stats :as ms]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.protocols :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defprotocol PNeuralNetwork
  (init [this])
  (set-trainable [this trainable])
  (add [this layer])
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
  (init [this]
        (assoc this
               :val-set (when validation-data
                          {:X (validation-data 0)
                           :y (validation-data 1)})))

  (set-trainable [this trainable]
                 (map #(assoc %
                              :trainable trainable)
                      layers))

  (add [this layer]
       (update this
               conj
               :layers (initialize (if (seq layers)
                                     ;; If this is not the first layer added then set the input shape
                                     ;; to the output shape of the last added layer
                                     (set-input-shape layer
                                                      (output-shape (last layers)))
                                     layer))))

  (test-on-batch [this X y]
                 (let [y-pred (:output (forward-pass* X false))
                       loss (ms/mean (loss loss-function y y-pred))
                       acc (acc loss-function y y-pred)]
                   [loss acc]))

  (train-on-batch [this X y]
                  (let [y-pred (forward-pass* this X true)
                        loss (ms/mean (loss loss-function y y-pred))
                        acc (acc loss-function y y-pred)
                        loss-grad (grad loss-function y y-pred)]
                    [(backward-pass* this loss-grad) loss acc]))

  (forward-pass* [this X training]
                 (reduce #(forward-pass %2 %1 training)
                         X
                         layers))

  (backward-pass* [this loss-grad]
                  (reduce #(backward-pass %2 %1)
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
                 (prn table-data)
                 (prn (str "Total Parameters: \n" tot-params)))
               (let [layer-name (layer-name layer)
                     params (parameters layer)
                     out-shape (output-shape layer)]
                 (recur n
                        (conj table-data
                              [layer-name (str params) (str out-shape)])
                        (+ tot-params params))))))

  PModel
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
                                                ((train-on-batch X-batch
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
                                              ((test-on-batch (:X val-set)
                                                              (:y val-set)) 0))
                                        (:validation errors))))))))

  (predict [this X]
           (forward-pass* this X false)))
