(ns mlalgorithms.deep-learning.layers
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix.stats :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.stats :as ms]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defprotocol Layer
  (initialize [this optimizer])
  (set-input-shape [this shape])
  (layer-name [this])
  (parameters [this])
  (forward-pass [this X training])
  (backward-pass [this accum-grad])
  (output-shape [this]))

(defrecord Dense
    [n-units input-shape
     layer-input trainable
     W w0 W-opt w0-opt
     accum-grad]
  Layer
  (initialize [this optimizer]
    (let [limit (/ 1 (sqrt (input-shape 0)))]
      (assoc this
             :W (m/uniform (- limit)
                           limit
                           [(input-shape 0) n-units])
             :w0 (m/zeros [1 n-units])
             :W-opt optimizer
             :w0-opt optimizer)))

  (parameters [this]
    (+ (m/prod (shape W)) (m/prod (shape w0))))

  (forward-pass [this X training]
    (assoc this
           ;; same as (+ (dot X (:W this)) (:w0 this))
           :W (+ (dot X W) w0)
           :layer-input X))

  (backward-pass [this accum-grad]
    (merge this
           (when trainable
             {:W (update W-opt
                         W
                         (dot (transpose layer-input)
                              accum-grad))
              :w0 (update w0-opt
                          w0
                          ;; keepdims
                          [(ms/sum accum-grad)])})
           ;; Return accumulated gradient for next layer
           ;; Calculated based on the weights used during the forward pass
           {:accum-grad (dot accum-grad (transpose W))}))

  (output-shape [this]
    [n-units]))
