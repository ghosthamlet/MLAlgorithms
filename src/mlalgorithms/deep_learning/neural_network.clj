(ns mlalgorithms.deep-learning.neural-network
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.stats :as ms]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.selection :as sel]
            [clojure.pprint :as pp]
            [clojure.string :as s]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.deep-learning.layers :as layer]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defn get-last-layer-output [nn]
  (->> nn :layers last :output))

(defn get-type-name [x]
  (last (s/split (str (type x)) #"\.")))

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
   (errors {:training [] :validation []}) (val-set)]
  PNeuralNetwork
  (init-nn [this]
           (assoc this
                  :val-set (when validation-data
                             {:X (validation-data 0)
                              :y (validation-data 1)})))

  (set-trainable [this trainable]
                 (assoc this
                        :layers (map #(assoc %
                                             :trainable trainable)
                                     layers)))

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
                 (alog)
                 (alog "test-on-batch")
                 (let [y-pred (get-last-layer-output (forward-pass* this X false))
                       _ (alog (get-type-name loss-function))
                       loss (m/mean (p/loss loss-function y y-pred))
                       acc (p/acc loss-function y y-pred)]
                   [loss acc]))

  (train-on-batch [this X y]
                  (alog)
                  (alog "train-on-batch")
                  (let [nn (forward-pass* this X true)
                        y-pred (get-last-layer-output nn)
                        _ (alog (get-type-name loss-function))
                        loss (m/mean (p/loss loss-function y y-pred))
                        acc (p/acc loss-function y y-pred)
                        loss-grad (p/loss-grad loss-function y y-pred)]
                    [(backward-pass* nn loss-grad) loss acc]))

  (forward-pass* [this X training]
                 (alog)
                 (alog "forward-pass*")
                 (:nn (reduce #(let [_ (alog (format "*** %s ***" (get-type-name %2)))
                                     layer (layer/forward-pass %2 (:output %1) training)]
                                 (assoc (update-in %1 [:nn :layers] conj layer)
                                        :output (:output layer)))
                              {:nn (assoc this :layers [])
                               :output X}
                              layers)))

  (backward-pass* [this loss-grad]
                  (alog)
                  (alog "backward-pass*")
                  (update-in (:nn (reduce #(let [_ (alog (format "*** %s ***" (get-type-name %2)))
                                                 layer (layer/backward-pass %2 (:accum-grad %1))]
                                             (assoc (update-in %1 [:nn :layers] conj layer)
                                                    :accum-grad (:accum-grad layer)))
                                          {:nn (assoc this :layers [])
                                           :accum-grad loss-grad}
                                          (reverse layers)))
                             [:layers]
                             reverse))

  (summary [this name]
           (println (str "Model: " name))
           (println (str "Input Shape: " (:input-shape (layers 0))))
           (loop [[layer & n] layers
                  table-data [["Layer Type" "Parameters" "Output Shape"]]
                  tot-params 0]
             (if (nil? layer)
               (do
                 (pp/pprint table-data)
                 (println (str "Total Parameters: \n" tot-params)))
               (let [layer-name (str (type layer))
                     params (layer/parameters layer)
                     out-shape (layer/output-shape layer)]
                 (recur n
                        (conj table-data
                              [layer-name (str params) (str out-shape)])
                        (+ tot-params params))))))

  p/PModel
  (fit [this X y n-epochs batch-size]
       (let [n-samples ((shape X) 0)
             batch-error-fn #(loop [[i & is] (range 0 n-samples batch-size)
                                    batch-error []]
                               (if (nil? i)
                                 batch-error
                                 (let [[begin end] [i (min (+ i batch-size)
                                                           n-samples)]
                                       X-batch (sel/sel X (sel/irange begin (dec end)) (sel/irange) (sel/irange))
                                       y-batch (sel/sel y (sel/irange begin (dec end)) (sel/irange) (sel/irange))]
                                   (recur is
                                          (conj batch-error
                                                ((train-on-batch this
                                                                 X-batch
                                                                 y-batch)
                                                 1))))))]
         (loop [[i & is] (range n-epochs)
                -errors errors]
           (if (nil? i)
             (assoc this
                    :errors -errors)
             (recur is
                    (assoc -errors
                           :training (conj (:training -errors)
                                           (m/mean (batch-error-fn)))
                           :validation (if val-set
                                         (conj (:validation -errors)
                                               ((test-on-batch this
                                                               (:X val-set)
                                                               (:y val-set)) 0))
                                         (:validation -errors))))))))

  (predict [this X]
           (alog)
           (alog "predict")
           (get-last-layer-output (forward-pass* this X false))))
