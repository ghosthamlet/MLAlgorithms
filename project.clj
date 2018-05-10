(defproject mlalgorithms "0.1.0-SNAPSHOT"
  :description "use closure and return fn to capsurize init params or states like class.
                keyword and multimethod for activiaton loss grad fns. "
  :url "https://github.com/ghosthamlet/MLAlgorithms"
  :license  {:name "Eclipse Public License"
             :url "http://www.eclipse.org/legal/epl-v10.html"}
  ;; https://github.com/technomancy/leiningen/blob/stable/doc/PROFILES.md
  :plugins [[lein-kibit "0.1.6"]
            ;; lein cljfmt check/fix
            [lein-cljfmt "0.5.7"]]
  :dependencies  [[org.clojure/clojure "1.9.0"]
                  [net.mikera/core.matrix "0.62.0"]
                  [net.mikera/vectorz-clj "0.47.0"]
                  ;; https://github.com/metasoarous/oz
                  [metasoarous/oz "1.3.1"]

                  [org.thnetos/cd-client "0.3.6"]
                  ;; JUST use *e or (.printStackTrace *e), no need below many debug tools
                  [org.clojure/tools.trace "0.7.9"]
                  ;; https://github.com/gfredericks/debug-repl
                  [com.gfredericks/debug-repl "0.0.9"]
                  ;; http://bpiel.github.io/sayid/
                  [com.billpiel/sayid "0.0.16"]
                  ;; https://github.com/dgrnbrg/spyscope
                  [spyscope "0.1.6"]
                  ;; https://github.com/vvvvalvalval/scope-capture
                  [vvvvalvalval/scope-capture "0.1.4"]
                  [philoskim/debux-stubs "0.4.7"]]
  ;; (require '[clojure.reflect :as reflect])
  ;; (pp/pprint (ancestors (type str)))
  ;; (->> (reflect/reflect java.io.InputStream) :members (sort-by :name) (pp/print-table [:name :flags :parameter-types :return-type]))
  :repl-options {:nrepl-middleware
                 [com.gfredericks.debug-repl/wrap-debug-repl]}
  :profiles {:dev {:dependencies [[philoskim/debux "0.4.7"]]}})
