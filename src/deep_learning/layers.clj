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

(declare determine-padding
         get-im2col-indices
         image-to-column
         column-to-image)

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

(defrecord Conv2D
    [n-filters filter-shape
     input-shape padding stride
     layer-input trainable
     X-col W-col outputs
     W w0 W-opt w0-opt
     accum-grad]
  Layer
  (initialize [this optimizer]
    (let [[filter-height filter-width] filter-shape
          channels (input-shape 0)
          limit (/ 1 (sqrt (m/prod filter-shape)))]
      (assoc this
             :W (m/uniform (- limit)
                           limit
                           [n-filters channels filter-height filter-width])
             ;; FIXME: zeros param
             :w0 (m/zeros n-filters 1)
             :W-opt optimizer
             :w0-opt optimizer)))

  (parameters [this]
    (+ (m/prod (shape W)) (m/prod (shape w0))))

  (forward-pass [this X training]
    (let [[batch-size channels height width] (shape X)
          ;; Turn image shape into column shape 
          ;; (enables dot product between input and weights)
          X-col* (image-to-column X filter-shape stride padding)
          ;; Turn weights into column shape
          W-col* (reshape W [n-filters -1])
          ;; Calculate output
          output (+ (dot W-col* X-col*) w0)
          ;; Reshape into (n_filters, out_height, out_width, batch_size)
          output (reshape output (+ (output-shape this) [batch-size]))]
      (assoc this
             :layer-input X
             :X-col X-col*
             :W-col W-col*
             ;; Redistribute axises so that batch size comes first
             :outputs (transpose output 3 0 1 2))))

  (backward-pass [this accum-grad]
    (let [accum-grad (reshape (transpose accum-grad [1 2 3 0]) n-filters -1) ;; Reshape accumulated gradient into column shape
          ;; Recalculate the gradient which will be propogated back to prev. layer
          accum-grad (dot (transpose W-col) accum-grad)
          ;; Reshape from column shape to image shape
          accum-grad (column-to-image accum-grad
                                      (shape layer-input)
                                      filter-shape
                                      stride
                                      padding)]
      (merge this
             (when trainable
               ;; FIXME: reshape param
               ;; Take dot product between column shaped accum. gradient and column shape
               ;; layer input to determine the gradient at the layer with respect to layer weights
               (let [grad-w (reshape (dot accum-grad (transpose X-col))
                                     (shape W))
                     ;; FIXME: sum param
                     ;; The gradient with respect to bias terms is the sum similarly to in Dense layer
                     grad-w0 (sum accum-grad 1 true)
                     {:W (update W-opt W grad-w)
                      :w0 (update w0-opt w0 grad-w0)}]))
             {:accum-grad accum-grad})))
  (output-shape [this]
    (let [[channels height width] input-shape
          [pad-h padw] (determine-padding filter-shape padding)
          output-height (-> height
                            (+ (sum pad-h))
                            (- (filter-shape 0))
                            (/ stride)
                            (+ 1))
          output-width (-> width
                           (+ (sum pad-w))
                           (- (filter-shape 1))
                           (/ stride)
                           (+ 1))]
      [n-filters (int output-height) (int output-width)])))





;; Method which calculates the padding based on the specified output shape and the
;; shape of the filters
(defn determine-padding
  ([filter-shape]
   (determine-padding filter-shape "same"))
  ([filter-shape output-shape]
   (if (= "valid" output-shape)
     ;; No padding
     [[0 0] [0 0]]
     ;; Pad so that the output shape is the same as input shape (given that stride=1)
     (let [[filter-height filter-width] filter-shape
           ;; Derived from:
           ;; output_height = (height + pad_h - filter_height) / stride + 1
           ;; In this case output_height = height and stride = 1. This gives the
           ;; expression for the padding below.
           pad-fn #(-> %1 (- 1) (/ 2) %2 int)
           pad-h1 (pad-fn filter-height floor)
           pad-h2 (pad-fn filter-height ceil)
           pad-w1 (pad-fn filter-width floor)
           pad-w2 (pad-fn filter-width ceil)]
       [[pad-h1 pad-h2] [pad-w1 pad-w2]]))))

(defn get-im2col-indices
  ([images-shape filter-shape padding]
   (get-im2col-indices images-shape filter-shape padding 1))
  ([images-shape filter-shape padding stride]
   (let [[batch-size channels height width] images-shape
         [filter-height filter-width] filter-shape
         [pad-h pad-w] padding
         out-fn #(-> %1 (+ (sum %2)) (- %3) (/ stride) (+ 1) int)
         out-height (out-fn height pad-h filter-height)
         out-width (out-fn width pad-w filter-width)
         i0 (repeat (range filter-height) filter-width)
         i0 (tile i0 channels)
         i1 (* stride (repeat (range out-height) out-width))
         j0 (tile (range filter-width) (* filter-height channels))
         j1 (* stride (tile (range out-width) out-height))
         i (+ (reshape i0 -1 1)
              (reshape i1 1 -1))
         j (+ (reshape j0 -1 1)
              (reshape j1 1 -1))
         k (reshape (repeat (range channels)
                            (* filter-height filter-width)) -1 1)]
     [k i j])))

;; Method which turns the image shaped input to column shape.
;; Used during the forward pass.
;; Reference: CS231n Stanford
(defn image-to-column
  ([images filter-shape stride]
   (image-to-column images filter-shape stride "same"))
  ([images filter-shape stride output-shape]
   (let [[filter-height filter-width] filter-shape
         [pad-h pad-w] (determine-padding filter-shape output-shape)
         ;; Add padding to the image
         images-padded (pad images [[0 0] [0 0] pad-h pad-w] "constant")
         ;; Calculate the indices where the dot products are to be applied between weights
         ;; and the image
         [k i j] (get-im2col-indices (shape images) filter-shape [pad-h pad-w] stride)
         ;; Get content from image at those indices
         cols (sel/sel images-padded (sel/irange) k i j)
         channels ((shape images) 1)
         ;; Reshape content into column shape
         cols (reshape (transpose cols 1 2 0) (* filter-height filter-width channels) -1)]
     cols)))

;; Method which turns the column shaped input to image shape.
;; Used during the backward pass.
;; Reference: CS231n Stanford
(defn column-to-image
  ([cols images-shape filter-shape stride]
   (column-to-image cols images-shape filter-shape stride "same"))
  ([cols images-shape filter-shape stride output-shape]
   (let [[batch-size channels height width] images-shape
         [pad-h pad-w] (determine-padding filter-shape output-shape)
         height-padded (+ height (sum pad-h))
         width-padded (+ width (sum pad-w))
         images-padded (m/empty [batch-size channels height-padded width-padded])
         ;; Calculate the indices where the dot products are applied between weights
         ;; and the image
         [k i j] (get-im2col-indices images-shape filter-shape [pad-h pad-w] stride)
         cols (reshape cols (* channels (m/prod filter-shape)) -1 batch-size)
         cols (transpose cols 2 0 1)
         ;; FIXME: fn np.add.at and slice
         ;; Add column content to the images at the indices
         images-padded (np.add.at images-padded [(slice nil) k i j] cols)]
     ;; Return image without padding
     (sel/sel images-padded
              (sel/irange)
              (sel/irange)
              (sel/irange (pad-h 0) (+ height (pad-h 0)))
              (sel/irange (pad-w 0) (+ width (pad-w 0)))))))
