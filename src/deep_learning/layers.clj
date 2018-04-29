(ns mlalgorithms.deep-learning.layers
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix.stats :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.stats :as ms]
            [clojure.core.matrix.selection :as sel]
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

(defrecord RNN
    [n-units input-shape
     layer-input trainable
     activation bptt-trunc
     state-input states outputs
     U V W U-opt V-opt W-opt
     accum-grad]
  Layer
  (initialize [this optimizer]
    (let [[timesteps input-dim] input-shape
          ulimit (/ 1 (sqrt input-dim))
          vlimit (/ 1 (sqrt n-units))]
      (assoc this
            :U (m/uniform (- ulimit) ulimit [n-units input-dim])
            :V (m/uniform (- vlimit) vlimit [input-dim n-units])
            :W (m/uniform (- vlimit) vlimit [n-units n-units])
            :U-opt optimizer
            :V-opt optimizer
            :W-opt optimizer
            :activation ((activation activation_functions))
            :trainable true)))

  (parameters [this]
    (+ (m/prod (shape U))
       (m/prod (shape V))
       (m/prod (shape W))))

  (forward-pass [this X _]
    (let [[batch-size timesteps input-dim] (shape X)]
      (loop [[t & ts] (range timesteps)
             ;; Save these values for use in backprop.
             state-input (m/zeros [batch-size timesteps n-units])
             states (m/zeros [batch-size (inc timesteps) n-units])
             outputs (m/zeros [batch-size timesteps input-dim])
             ;; Set last time step to zero for calculation of the state_input at time step zero
             states (sel/set-sel states (sel/irange) -1 (m/zeros [batch-size n-units]))]
        (if (empty? ts)
          (assoc this
                 :layer-input X
                 :state-input state-input
                 :states states
                 :outputs outputs)
          ;; Input to state_t is the current input and output of previous states
          (recur ts
                 (m/sety state-input
                         t
                         (+ (dot (m/gety X t) (transpose U))
                            (dot (m/gety states (dec t)) (transpose W))))
                 (m/sety states
                         t
                         (activation (m/gety state-input t)))
                 (m/sety outputs
                         t
                         (dot (m/gety states t) (transpose V))))))))

  (backward-pass [this accum-grad]
    (let [[_ timesteps _] (shape accum-grad)
          f (fn [[t* & ts*] grad-U grad-W grad-wrt-state]
              (if (empty? ts*)
                [grad-U grad-W grad-wrt-state]
                (recur ts*
                       (+ grad-U (dot (transpose grad-wrt-state)
                                      (m/gety layer-input t*)))
                       (+ grad-W (dot (transpose grad-wrt-state)
                                      (m/gety states (dec t*))))
                       ;; Calculate gradient w.r.t previous state
                       (* (dot grad-wrt-state W)
                          (grad activation (m/gety state-input (dec t*)))))))]
      ;; Back Propagation Through Time
      (loop [[t & ts] (reverse (range timesteps))
             ;; Variables where we save the accumulated gradient w.r.t each parameter
             grad-U (m/zeros-like U)
             grad-V (m/zeros-like V)
             grad-W (m/zeros-like W)
             ;; The gradient w.r.t the layer input.
             ;; Will be passed on to the previous layer in the network
             accum-grad-next (m/zeros-like accum-grad)]
          (if (empty? ts)
            (assoc this
                   :U (update U-opt U grad-U)
                   :V (update V-opt V grad-V)
                   :W (update W-opt W grad-W)
                   :accum-grad accum-grad-next)
            (let [grad-wrt-state (* (dot (m/gety accum-grad t) V)
                                    (grad activation (m/gety state-input t))) ;; Calculate the gradient w.r.t the state input
                  ;; Gradient w.r.t the layer input
                  accum-grad-next (m/sety accum-grad-next t (dot grad-wrt-state U))
                  ;; Update gradient w.r.t W and U by backprop. from time step t for at most
                  ;; self.bptt_trunc number of time steps
                  [grad-U grad-W grad-wrt-state] (f (->> t
                                                         inc
                                                         ;; in python np.arange
                                                         (->> bptt-trunc (- t) (max 0) range)
                                                         reverse)
                                                    grad-U
                                                    grad-W
                                                    grad-wrt-state)]
              (recur ts
                     grad-U
                     ;; Update gradient w.r.t V at time step t
                     (+ grad-V
                        (dot (transpose (m/gety accum-grad t))
                             (m/gety states t)))
                     grad-W
                     grad-wrt-state))))))

  (output-shape [this]
    input-shape))
