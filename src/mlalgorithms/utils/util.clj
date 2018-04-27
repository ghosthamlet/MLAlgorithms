(ns mlalgorithms.utils.util
  (:require [clojure.string :as s]))

(defmacro gen-record
  "
  (gen-record RNN [alpha (lr 0) w]
      Layer
      (forword [])

      Model
      (fit []))
  "
  [name fields & specs]
  (let [keys []
        as []]
    `(do
       (defrecord ~name ~keys
         ~@specs)

       (defn ~(symbol (str "make-" (s/lower-case name)))
         [& {:keys ~keys
             :as ~as}]
         (~(symbol (str name ".")) ~keys)))))

