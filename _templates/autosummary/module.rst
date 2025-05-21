{{ name | escape | underline}}

.. currentmodule:: {{ module }}
.. automodule:: {{ fullname }}

   {% block classes %}
   {%- if classes %}
      {% for item in classes %}
      .. autoclass:: {{item}}
         :members:
         :member-order: bysource
      {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}
   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block modules %}
   {%- if modules %}
   .. autosummary::
      :toctree:
      :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}