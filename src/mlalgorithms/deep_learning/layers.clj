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
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(declare activation-functions
         determine-padding
         get-im2col-indices
         image-to-column
         column-to-image)

;; tmp
(declare grad)

(defprotocol Layer
  (init [this])
  (initialize [this optimizer])
  (set-input-shape [this shape])
  (layer-name [this])
  (parameters [this])
  (forward-pass [this X training])
  (backward-pass [this accum-grad])
  (output-shape [this]))

(defpyrecord Dense
  [n-units (input-shape)
   (layer-input) (trainable true)
   (W) (w0) (W-opt) (w0-opt)
   (accum-grad)]
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

  (output-shape [this] [n-units]))

(defpyrecord RNN
  [n-units (input-shape)
   (layer-input) (trainable true)
   (activation "tanh") (bptt-trunc 5)
   (state-input) (states) (outputs)
   (U) (V) (W) (U-opt) (V-opt) (W-opt)
   (accum-grad)]
  Layer
  (initialize [this optimizer]
              (let [[timesteps input-dim] input-shape
                    ulimit (/ 1 (sqrt input-dim))
                    vlimit (/ 1 (sqrt n-units))]
                (assoc this
                       ;; Weight of the input
                       :U (m/uniform (- ulimit) ulimit [n-units input-dim])
                       ;; Weight of the output
                       :V (m/uniform (- vlimit) vlimit [input-dim n-units])
                       ;; Weight of the previous state
                       :W (m/uniform (- vlimit) vlimit [n-units n-units])
                       :U-opt optimizer
                       :V-opt optimizer
                       :W-opt optimizer
                       :activation ((activation activation-functions))
                       :trainable true)))

  (parameters [this]
              (+ (m/prod (shape U))
                 (m/prod (shape V))
                 (m/prod (shape W))))

  (forward-pass [this X _]
                (let [[batch-size timesteps input-dim] (shape X)
                      states (m/zeros [batch-size (inc timesteps) n-units])]
                  (loop [[t & ts] (range timesteps)
                         ;; Save these values for use in backprop.
                         state-input (m/zeros [batch-size timesteps n-units])
                         outputs (m/zeros [batch-size timesteps input-dim])
                         ;; Set last time step to zero for calculation of the state_input at time step zero
                         states (sel/set-sel states
                                             (sel/irange)
                                             -1
                                             (m/zeros [batch-size n-units]))]
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
                                               (grad activation
                                                     (m/gety state-input t))) ;; Calculate the gradient w.r.t the state input
                             ;; Gradient w.r.t the layer input
                             accum-grad-next (m/sety accum-grad-next
                                                     t
                                                     (dot grad-wrt-state U))
                             ;; Update gradient w.r.t W and U by backprop. from time step t for at most
                             ;; self.bptt_trunc number of time steps
                             [grad-U grad-W grad-wrt-state] (f (->> t
                                                                    inc
                                                                    ;; in python np.arange
                                                                    ;; FIXME: failed nested ->>
                                                                    ;; (->> bptt-trunc (- t) (max 0) range)
                                                                    (range (max 0 (- t bptt-trunc)))
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

  (output-shape [this] input-shape))

(defpyrecord Conv2D
  [n-filters filter-shape
   (input-shape) (padding "same") (stride 1)
   (layer-input) (trainable true)
   (X-col) (W-col) (outputs)
   (W) (w0) (W-opt) (w0-opt)
   (accum-grad)]
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
                      X-col* (image-to-column X
                                              filter-shape
                                              stride
                                              :output-shape padding)
                      ;; Turn weights into column shape
                      W-col* (m/reshape W [n-filters -1])
                      ;; Calculate output
                      output (+ (dot W-col* X-col*) w0)
                      ;; Reshape into (n_filters, out_height, out_width, batch_size)
                      output (m/reshape output
                                        (+ (output-shape this) [batch-size]))]
                  (assoc this
                         :layer-input X
                         :X-col X-col*
                         :W-col W-col*
                         ;; Redistribute axises so that batch size comes first
                         :outputs (transpose output [3 0 1 2]))))

  (backward-pass [this accum-grad]
                 (let [accum-grad (m/reshape (transpose accum-grad [1 2 3 0])
                                             [n-filters -1]) ;; Reshape accumulated gradient into column shape
                       ;; Recalculate the gradient which will be propogated back to prev. layer
                       accum-grad (dot (transpose W-col) accum-grad)
                       ;; Reshape from column shape to image shape
                       accum-grad (column-to-image accum-grad
                                                   (shape layer-input)
                                                   filter-shape
                                                   stride
                                                   :output-shape padding)]
                   (merge this
                          (when trainable
                            ;; Take dot product between column shaped accum. gradient and column shape
                            ;; layer input to determine the gradient at the layer with respect to layer weights
                            (let [grad-w (m/reshape (dot accum-grad (transpose X-col))
                                                    (shape W))
                                  ;; The gradient with respect to bias terms is the sum similarly to in Dense layer
                                  grad-w0 (m/sum accum-grad 1 true)]
                              {:W (update W-opt W grad-w)
                               :w0 (update w0-opt w0 grad-w0)}))
                          {:accum-grad accum-grad})))

  (output-shape [this]
                (let [[channels height width] input-shape
                      [pad-h pad-w] (determine-padding filter-shape
                                                       :output-shape padding)
                      output-fn #(-> %1
                                     (+ (ms/sum %2))
                                     (- (filter-shape 0))
                                     (/ stride)
                                     (+ 1))
                      output-height (output-fn height pad-h)
                      output-width (output-fn width pad-w)]
                  [n-filters (int output-height) (int output-width)])))

(defpyrecord BatchNormalization
  [(momentum 0.99) (trainable true)
   (eps 0.01) (running-mean)
   (running-var) (gamma)
   (beta) (gamma-opt)
   (beta-opt) (X-centered)
   (stddev-inv) (accum-grad)
   (output)]
  Layer
  (initialize [this optimizer]
              (assoc this
                     :gamma (m/ones input-shape)
                     :beta (m/zeros input-shape)
                     :gamma-opt optimizer
                     :beta-opt optimizer))

  (parameters [this]
              (+ (m/prod (shape gamma))
                 (m/prod (shape beta))))

  (forward-pass [this X training]
                (let [mean (ms/mean X)
                      var (ms/variance X)]
                  (merge this
                         ;; Initialize running mean and variance if first run
                         (when-not running-mean
                           {:running-mean mean
                            :running-var var})
                         (when (and training trainable)
                           {:running-mean (+ (* momentum running-mean)
                                             (* (- 1 momentum) mean))
                            :running-var (+ (* momentum running-var)
                                            (* (- 1 momentum) var))})
                         (let [X-centered (- X mean)
                               stddev-inv (/ 1 (sqrt (+ var eps)))
                               X-norm (* X-centered stddev-inv)]
                           ;; Statistics saved for backward pass
                           {:X-centered X-centered
                            :stddev-inv stddev-inv
                            :output (+ (* gamma X-norm) beta)}))))

  (backward-pass [this accum-grad]
                 (let []
                   (merge this
                          (when trainable
                            (let [X-norm (* X-centered stddev-inv)
                                  grad-gamma (m/sum (* accum-grad X-norm))
                                  grad-beta (m/sum accum-grad)]
                              {:gamma (update gamma-opt gamma grad-gamma)
                               :beta (update beta-opt beta grad-beta)}))
                          (let [batch-size ((shape accum-grad) 0)]
                            {:accum-grad (->> X-centered
                                              (* accum-grad)
                                              m/sum
                                              (* X-centered
                                                 (pow stddev-inv 2))
                                              (- (* batch-size accum-grad)
                                                 (m/sum accum-grad))
                                              (* (/ 1 batch-size)
                                                 gamma
                                                 stddev-inv))}))))

  (output-shape [this] input-shape))

(defpy pooling-forward [pooling-layer X (training true)]
  (let [[batch-size channels height width] (shape X)
        [_ out-height out-width] (output-shape pooling-layer)
        X* (m/reshape X [(* batch-size channels) 1 height width])
        X-col* (image-to-column X*
                                (:pool-shape pooling-layer)
                                (:stride pooling-layer)
                                (:padding pooling-layer))
        [cache output] (pooling-forward-pass pooling-layer X-col*)
        output (m/reshape output [out-height out-width batch-size channels])
        output (transpose output 2 3 0 1)]
    (assoc pooling-layer
          :layer-input X
          :cache cache
          :output output)))

(defpy pooling-backward [pooling-layer accum-grad]
  (let [[batch-size _ _ _] (shape accum-grad)
        [channels height width] (:input-shape pooling-layer)
        ;; np ravel
        accum-grad (m/ravel (transpose accum-grad 2 3 0 1))
        accum-grad-col (pooling-backward-pass pooling-layer accum-grad)
        accum-grad (column-to-image accum-grad-col [(* batch-size channels) 1 height width] (:pool-shape pooling-layer) (:stride pooling-layer) 0)
        accum-grad (m/reshape accum-grad (+ [batch-size] (:input-shape pooling-layer)))]
    (assoc pooling-layer
           :accum-grad accum-grad)))

(defpy pooling-output-shape [pooling-layer]
  (let [[channels height width] (:input-shape pooling-layer)
        out-fn #(+ (/ (- %1 ((:pool-shape pooling-layer) %2))
                      (:stride pooling-layer))
                   1)
        out-height (out-fn height 0)
        out-width (out-fn width 1)]
    (assert! (= (% out-height 1) 0))
    (assert! (= (% out-width 1) 0))
    [channels (int out-height) (int out-width)]))

(defprotocol PoolingLayer
  (pooling-forward-pass [this X-col])
  (pooling-backward-pass [this accum-grad]))

(defpyrecord MaxPooling2D
  [(pool-shape [2 2]) (stride 1)
   (padding 0) (trainable true)
   (layer-input) (accum-grad)
   (input-shape) (cache)]
  Layer
  (forward-pass [this X trainable]
                (pooling-forward this X :trainable trainable))

  (backward-pass [this accum-grad]
                 (pooling-backward this accum-grad))

  (output-shape [this]
                (pooling-output-shape this))

  PoolingLayer
  (pooling-forward-pass [this X-col]
                        (let [arg-max (m/flatten (m/argmax X-col))
                              output (sel/sel arg-max (range (size arg-max)))]
                          [arg-max output]))

  (pooling-backward-pass [this accum-grad]
                         (let [accum-grad-col (m/zeros [(m/prod pool-shape)
                                                        (size accum-grad)])
                               arg-max cache]
                           (sel/set-sel accum-grad-col
                                        arg-max
                                        (range (size accum-grad))
                                        accum-grad))))

(defpyrecord AveragePooling2D
  [(pool-shape [2 2]) (stride 1)
   (padding 0) (trainable true)
   (layer-input) (accum-grad)
   (input-shape) (cache)]
  Layer
  (forward-pass [this X trainable]
                (pooling-forward this X :trainable trainable))

  (backward-pass [this accum-grad]
                 (pooling-backward this accum-grad))

  (output-shape [this]
                (pooling-output-shape this))

  PoolingLayer
  (pooling-forward-pass [this X-col]
                        (let [output (ms/mean X-col)]
                          [cache output]))

  (pooling-backward-pass [this accum-grad]
                         (let [accum-grad-col (m/zeros [(m/prod pool-shape)
                                                        (size accum-grad)])]
                           (m/sety accum-grad-col
                                        (range (size accum-grad))
                                        (* (/ 1. ((shape accum-grad-col) 0))
                                           accum-grad)))))

(defpy padding-forward [padding-layer X (training true)]
  (assoc padding-layer
         :output (m/pad X
                        [[0 0] [0 0] (padding 0) (padding 1)]
                        :mode "constant"
                        :constant-values (:padding-value padding-layer))))

(defpy padding-backward [padding-layer accum-grad]
  (let [[pad-top pad-left] [(get-in (:padding padding-layer) [0 0])
                            (get-in (:padding padding-layer) [1 0])]
        [height width] [((:input-shape padding-layer) 1)
                        ((:input-shape padding-layer) 2)]
        accum-grad (sel/sel accum-grad
                            (sel/irange)
                            (sel/irange)
                            (sel/irange pad-top (+ pad-top height))
                            (sel/irange pad-left (+ pad-left width)))]
    accum-grad))

(defpy padding-output-shape [padding-layer]
  (let [new-height (+ ((:input-shape padding-layer) 1)
                      (m/sum ((:padding padding-layer) 0)))
        new-width (+ ((:input-shape padding-layer) 2)
                     (m/sum ((:padding padding-layer) 1)))]
    [((:input-shape padding-layer) 0) new-height new-width]))

(defpyrecord ZeroPadding2D
  [(padding) (padding-value 0)
   (trainable true) (output)
   (accum-grad)]
  Layer
  (init [this]
        (let [padding (if (int? (padding 0))
                        [[(padding 0) (padding 0)] (padding 1)]
                        padding)
              padding (if (int? (padding 1))
                        [(padding 0) [(padding 1) (padding 1)]])]
          (assoc this
                 :padding padding
                 :padding-value 0)))

  (forward-pass [this X trainable]
                (padding-forward this X :trainable trainable))

  (backward-pass [this accum-grad]
                 (padding-backward this accum-grad))

  (output-shape [this]
                (padding-output-shape this)))

(defpyrecord Flatten
  [(input-shape) (prev-shape)
   (trainable true) (output)
   (accum-grad)]
  Layer
  (forward-pass [this X training]
                (assoc this
                       :prev-shape (shape X)
                       :output (reshape X [((shape X) 0) -1])))

  (backward-pass [this accum-grad]
                 (assoc this
                        :accum-grad (reshape accum-grad prev-shape)))

  (output-shape [this]
                [(m/prod input-shape)]))

(defpyrecord UpSampling2D
  [(size [2 2]) (input-shape)
   (prev-shape) (trainable true)
   (output) (accum-grad)]
  Layer
  (forward-pass [this X training]
                (assoc this
                       :prev-shape (shape X)
                       :output (m/repeat (m/repeat X (size 0) :axis 2)
                                         (size 1)
                                         :axis 3)))

  (backward-pass [this accum-grad]
                 (assoc this
                        :accum-grad "np: accum_grad[:, :, ::self.size[0], ::self.size[1]]"))

  (output-shape [this]
                (let [[channels height width] input-shape]
                  [channels (* (size 0) height) (* (size 1) width)])))

(defpyrecord Reshape
  [shape (input-shape)
   (prev-shape) (trainable true)
   (output) (accum-grad)]
  Layer
  (forward-pass [this X training]
                (assoc this
                       :prev-shape (shape X)
                       :output (reshape X (+ [((shape X) 0)] shape))))

  (backward-pass [this accum-grad]
                 (assoc this
                        :accum-grad (reshape accum-grad prev-shape)))

  (output-shape [this] shape))

(defpyrecord Dropout
  [(p 0.2) (mask*)
   (input-shape) (n-units)
   (pass-through) (trainable)
   (output) (accum-grad)]
  Layer
  (forward-pass [this X training]
                (let [c (- 1 p)]
                  (merge this
                         (if training
                           (let [mask* (> (m/uniform (shape X)) p)]
                             {:mask* mask*
                              :output (* X mask*)})
                           {:output (* X c)}))))

  (backward-pass [this accum-grad] (* accum-grad mask*))

  (output-shape [this] input-shape))

(def activation-functions
  {:relu a/ReLU
   :sigmoid a/Sigmoid
   :selu a/SELU
   :elu a/ELU
   :softmax a/Softmax
   :leaky-relu a/LeakyReLU
   :tanh a/TanH
   :softplus a/SoftPlus
   })

(defpyrecord Activation
  [activation-func (trainable true)
   (layer-input) (output)
   (accum-grad)]
  Layer
  (forward-pass [this X training]
                (assoc this
                       :layer-input X
                       :output (activation-func X)))

  (backward-pass [this accum-grad]
                 (assoc this
                        :accum-grad (* accum-grad
                                       (grad activation-func layer-input))))

  (output-shape [this] input-shape))

;; Method which calculates the padding based on the specified output shape and the
;; shape of the filters
(defpy determine-padding
  [filter-shape (output-shape "same")]
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
      [[pad-h1 pad-h2] [pad-w1 pad-w2]])))

(defpy get-im2col-indices
  [images-shape filter-shape padding (stride 1)]
  (let [[batch-size channels height width] images-shape
        [filter-height filter-width] filter-shape
        [pad-h pad-w] padding
        out-fn #(-> %1
                    (+ (ms/sum %2))
                    (- %3)
                    (/ stride)
                    (+ 1)
                    int)
        out-height (out-fn height
                           pad-h
                           filter-height)
        out-width (out-fn width
                          pad-w
                          filter-width)
        i0 (m/repeat (range filter-height)
                     filter-width)
        i0 (m/tile i0 channels)
        i1 (* stride
              (m/repeat (range out-height) out-width))
        j0 (m/tile (range filter-width)
                   (* filter-height channels))
        j1 (* stride
              (m/tile (range out-width) out-height))
        i (+ (m/reshape i0 [-1 1])
             (m/reshape i1 [1 -1]))
        j (+ (m/reshape j0 [-1 1])
             (m/reshape j1 [1 -1]))
        k (m/reshape (m/repeat (range channels)
                               (* filter-height filter-width))
                     [-1 1])]
    [k i j]))

;; Method which turns the image shaped input to column shape.
;; Used during the forward pass.
;; Reference: CS231n Stanford
(defpy image-to-column
  [images filter-shape stride (output-shape "same")]
  (let [[filter-height filter-width] filter-shape
        [pad-h pad-w] (determine-padding filter-shape
                                         :output-shape output-shape)
        ;; Add padding to the image
        images-padded (m/pad images
                             [[0 0] [0 0] pad-h pad-w]
                             :mode "constant")
        ;; Calculate the indices where the dot products are to be applied between weights
        ;; and the image
        [k i j] (get-im2col-indices (shape images)
                                    filter-shape
                                    [pad-h pad-w]
                                    :stride stride)
        ;; Get content from image at those indices
        cols (sel/sel images-padded (sel/irange) k i j)
        channels ((shape images) 1)
        ;; Reshape content into column shape
        cols (m/reshape (transpose cols [1 2 0])
                        [(* filter-height filter-width channels) -1])]
    cols))

;; Method which turns the column shaped input to image shape.
;; Used during the backward pass.
;; Reference: CS231n Stanford
(defpy column-to-image
  [cols images-shape filter-shape stride (output-shape "same")]
  (let [[batch-size channels height width] images-shape
        [pad-h pad-w] (determine-padding filter-shape
                                         :output-shape output-shape)
        height-padded (+ height (ms/sum pad-h))
        width-padded (+ width (ms/sum pad-w))
        images-padded (m/empty [batch-size channels height-padded width-padded])
        ;; Calculate the indices where the dot products are applied between weights
        ;; and the image
        [k i j] (get-im2col-indices images-shape
                                    filter-shape
                                    [pad-h pad-w]
                                    :stride stride)
        cols (m/reshape cols [(* channels (m/prod filter-shape)) -1 batch-size])
        cols (transpose cols [2 0 1])
        ;; Add column content to the images at the indices
        images-padded (m/add-at images-padded
                                [(sel/irange) k i j]
                                cols)]
    ;; Return image without padding
    (sel/sel images-padded
             (sel/irange)
             (sel/irange)
             (sel/irange (pad-h 0) (+ height (pad-h 0)))
             (sel/irange (pad-w 0) (+ width (pad-w 0))))))
