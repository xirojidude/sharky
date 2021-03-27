
Shader "Skybox/FractalStarfield"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            #define R(p,a,r)lerp(a*dot(p,a),p,cos(r))+sin(r)*cross(p,a)

            fixed4 frag (v2f v) : SV_Target
            {
                fixed4 O = fixed4(0,0,0,0);
                float2 C = v.vertex;

                float3 p,r=float3(800.,480.,0.),
                d=normalize(float3((C-.5*r.xy)/r.y,1));  
                for(float i=0.,g,e,s;
                    ++i<99.;
                    O.xyz+=5e-5*abs(cos(float3(3,2,1)+log(s*9.)))/dot(p,p)/e
                )
                {
                    p=g*d;
                    p.z+=_Time.y*.1;
                    p=R(p,normalize(float3(1,2,3)),.5);   
                    s=2.5;
//                    p=abs(mod(p-1.,2.)-1.)-1.;
                    p=abs((p-1.)%2.-1.)-1.;
                    
                    for(int j=0;j++<10;)
                        p=1.-abs(p-float3(-1.,-1.,-1.)),
                        s*=e=-1.8/dot(p,p),
                        p=p*e-.7;
                        g+=e=abs(p.z)/s+.001;
                 }
                 O /= 4.0;

                return O;
            }




            ENDCG
        }
    }
}
