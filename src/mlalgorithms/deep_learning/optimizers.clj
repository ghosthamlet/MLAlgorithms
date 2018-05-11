(ns mlalgorithms.deep-learning.optimizers
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defpyrecord StochasticGradientDescent
  [(learning-rate 0.01) (momentum 0)
   (w-updt) (output)]
  p/POptimizer
  (opt-grad [this w grad-wrt-w]
               (let [w-updt (if w-updt
                              w-updt
                              (m/zeros (shape w)))
                     ;; Use momentum if set
                     w-updt (+ (* momentum w-updt)
                               (* (- 1 momentum) grad-wrt-w))]
                 (assoc this
                        :w-updt w-updt
                        ;; Move against the gradient to minimize loss
                        :output (- w (* learning-rate w-updt))))))

(defpyrecord NesterovAcceleratedGradient
  [(learning-rate 0.001) (momentum 0.4)
   (w-updt []) (output)]
  p/POptimizer
  (opt-grad [this w grad-func]
               ;; Calculate the gradient of the loss a bit further down the slope from w
               (let [approx-future-grad (m/clip (grad-func (- w (* momentum w-updt))) -1 1)
                     w-updt (if-not (m/any w-updt)
                              (m/zeros (shape w))
                              w-updt)
                     w-updt (+ (* momentum w-updt)
                               (* learning-rate approx-future-grad))]
                 (assoc this
                        :w-updt w-updt
                        ;; Move against the gradient to minimize loss
                        :output (- w w-updt)))))

(defpyrecord Adagrad
  [(learning-rate 0.01)
   (G) ; # Sum of squares of the gradients
   (eps 1e-8) (output)]
  p/POptimizer
  (opt-grad [this w grad-wrt-w]
               (let [G (if G G (m/zeros (shape w)))
                     ;; Add the square of the gradient of the loss function at w
                     G (+ G (pow grad-wrt-w 2))]
                 (assoc this
                        :G G
                        ;; Adaptive gradient with higher learning rate for sparse data
                        :output (- w
                                   (/ (* learning-rate grad-wrt-w)
                                      (sqrt (+ G eps))))))))

(defpyrecord Adadelta
  [(rho 0.95) (eps 1e-6)
   (E-w-updt) ; Running average of squared parameter updates
   (E-grad) ; Running average of the squared gradient of w
   (w-updt) (ouput)]
  p/POptimizer
  (opt-grad [this w grad-wrt-w]
               (let [[w-updt E-w-updt E-grad] (if w-updt
                                                [w-updt E-w-updt E-grad]
                                                [(m/zeros (shape w))
                                                 (m/zeros (shape w))
                                                 (m/zeros (shape grad-wrt-w))])
                     ;; Update average of gradients at w
                     E-grad (+ (* rho E-grad)
                               (* (- 1 rho)
                                  (pow grad-wrt-w 2)))
                     RMS-delta-w (sqrt (+ E-w-updt eps))
                     RMS-grad (sqrt (+ E-grad eps))
                     ;; Adaptive learning rate
                     adaptive-lr (/ RMS-delta-w RMS-grad)
                     ;; Calculate the update
                     w-updt (* adaptive-lr grad-wrt-w)
                     ;; Update the running average of w updates
                     E-w-updt (+ (* rho E-w-updt)
                                 (* (- 1 rho)
                                    (pow w-updt 2)))]
                 (assoc this
                        :w-updt w-updt
                        :E-w-updt E-w-updt
                        :E-grad E-grad
                        :output (- w w-updt)))))

(defpyrecord RMSprop
  [(learning-rate 0.01) (rho 0.9)
   (Eg) ; Running average of the square gradients at w
   (eps 1e-8) (output)]
  p/POptimizer
  (opt-grad [this w grad-wrt-w]
               (let [Eg (if Eg
                          Eg
                          (m/zeros (shape grad-wrt-w)))
                     Eg (+ (* rho Eg)
                           (* (- 1 rho)
                              (pow grad-wrt-w 2)))]
                 (assoc this
                        :Eg Eg
                        ;; Divide the learning rate for a weight by a running average of the magnitudes of recent
                        ;; gradients for that weight
                        :output (- w
                                   (/ (* learning-rate grad-wrt-w)
                                      (sqrt (+ Eg eps))))))))

(defpyrecord Adam
  [(learning-rate 0.001) (eps 1e-8)
   ;; Decay rates
   (b1 0.9) (b2 0.999)
   (m) (v)]
  p/POptimizer
  (opt-grad [this w grad-wrt-w]
               (let [[m v] (if m
                             [m v]
                             [(m/zeros (shape grad-wrt-w))
                              (m/zeros (shape grad-wrt-w))])
                     m (+ (* b1 m)
                          (* (- 1 b1)
                             grad-wrt-w))
                     v (+ (* b2 v)
                          (* (- 1 b2)
                             (pow grad-wrt-w 2)))
                     m-hat (/ m (- 1 b1))
                     v-hat (/ v (- 1 b2))
                     w-updt (/ (* learning-rate m-hat)
                               (+ (sqrt v-hat) eps))]
                 (assoc this
                        :m m
                        :v v
                        :w-updt w-updt
                        :output (- w w-updt)))))
