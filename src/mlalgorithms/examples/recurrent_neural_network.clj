(ns mlalgorithms.examples.recurrent-neural-network
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.selection :as sel]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.deep-learning.neural-network :as nn]
            [mlalgorithms.deep-learning.layers :as layer]
            [mlalgorithms.deep-learning.optimizers :as optimizer]
            [mlalgorithms.deep-learning.loss-functions :as loss]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]
            [oz.core :as oz]))

(defn- -gen-mult [nums n-col gen-fn]
  (loop [[i & is] (range nums)
         X (m/zeros [nums 10 n-col] :dtype :float)
         y (m/zeros [nums 10 n-col] :dtype :float)]
    (if (nil? i)
      [X (sel/set-sel y (sel/irange) (m/-x y 1) 1 1)]
      (let [mult (gen-fn)
            Xi (sel/sel X i)]
        (recur is
               (sel/set-sel X i (sel/irange) (sel/irange)
                            (to-categorical mult :n-col n-col))
               (sel/set-sel y i (sel/irange) (sel/irange)
                            (m/roll Xi (m/-x Xi 0) :axis 0)))))))

(defn- gen-mult-ser [nums]
  (-gen-mult nums
             61
             #(let [start (+ 2 (rand-int 5))]
                (m/linspace start (* start 10) :num 10 :dtype :int))))

(defn- gen-num-seq [nums]
  (-gen-mult nums
             20
             #(let [start (rand-int 10)]
                (range start (+ start 10)))))

(defn -main [& args]
  (let [optimizer (optimizer/make-adam)
        [X y] (gen-mult-ser 3000)
        [X-train X-test y-train y-test] (train-test-split X y :test-size 0.4)
        clf (-> (nn/make-neuralnetwork optimizer
                                       :loss-function (loss/make-crossentropy))
                (nn/add-layer (layer/make-rnn 10
                                              :activation :tanh
                                              :bptt-trunc 5
                                              :input-shape [10 61]))
                (nn/add-layer (layer/make-activation :softmax)))
        _ (nn/summary clf "RNN")
        tmp-X (m/argmax (sel/sel X-train 0) :axis 1)
        tmp-y (m/argmax (sel/sel y-train 0) :axis 1)
        _ (prn "Number Series Problem:")
        _ (prn "X = [" (s/join (map str tmp-X) "") "]")
        _ (prn "y = [" (s/join (map str tmp-y) "") "]")
        _ (prn)
        [train-err _] (nn/fit clf X-train y-train 500 512)
        y-pred (m/argmax (nn/predict X-test) :axis 2)
        y-test (m/argmax y-test :axis 2)
        accuracy (m/mean (accuracy-score y-test y-pred))
        _ (prn)
        _ (prn "Results:")]
    (dotimes [i 5]
      (let [tmp-X (m/argmax (sel/sel X-test i) :axis 1)
            tmp-y1 (sel/sel y-test i)
            tmp-y2 (sel/sel y-pred i)]
        (prn "X      = [" (s/join (map str tmp-X)) "]")
        (prn "y_true = [" (s/join (map str tmp-y1)) "]")
        (prn "y_pred = [" (s/join (map str tmp-y2)) "]")
        (prn)))
    (prn "Accuracy:" accuracy)
    (oz/start-plot-server!)
    (oz/v! {:data {:values (map-indexed (fn [i e] {:x i :y x}) train-err)}
            :encoding {:x {:field "x"
                           :axis {:title "Iterations"}}
                       :y {:field "y"
                           :axis {:title "Training Error"}}}
            :mark "line"
            :width 600})))
