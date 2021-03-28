
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

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

            #define R(p,a,r)lerp(a*dot(p,a),p,cos(r))+sin(r)*cross(p,a)

            fixed4 frag (v2f v) : SV_Target
            {
                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );

                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

                fixed4 O = fixed4(0,0,0,0);
                float2 C = v.vertex;

                float3 p=ro,r=rd, //  float3(800.,480.,0.),
                d=rd; //normalize(float3((C-.5*r.xy)/r.y,1));  
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
